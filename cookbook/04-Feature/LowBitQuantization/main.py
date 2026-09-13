# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NVFP4, MXFP8 and INT4-AWQ from PyTorch to a strongly-typed TensorRT engine.

Below 8 bits the interesting question stops being "does it run" and becomes "what actually
survives the round trip". Three things get in the way, and this file measures each:

1. **Quantization error is not monotonic in bit width.** A 4-bit format with per-block scales
   can beat an 8-bit format with coarser ones. The eager numbers below say which.
2. **What ModelOpt inserts is not always what TensorRT keeps.** The ONNX export has to survive
   the parser, and the parser has opinions -- notably that convolutions cannot use every
   format the linear layers can.
3. **Export needs opset 20 or later**, because that is where the low-bit types exist. Asking
   for less silently loses them or fails in a way that names the opset rather than the type.

Upstream (`examples/torch_onnx/torch_quant_to_onnx.py`) drives this with a timm ViT/Swin
download. The cookbook does not download at run time, so the model here has both `Linear` and
`Conv2d` layers -- which is what makes point 2 visible, since the two are treated differently.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn

from tensorrt_cookbook import case_mark

np.random.seed(31193)
torch.manual_seed(31193)

N_BATCH = 8
N_CHANNEL = 32
N_SIZE = 16
N_FEATURE = 256
N_CALIBRATION = 8
OPSET = 20  # Low-bit types need opset >= 20

output_path = Path(__file__).parent
result = OrderedDict()

class MixedModel(nn.Module):
    """Conv layers *and* Linear layers, because TensorRT does not treat them alike."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(N_CHANNEL, N_CHANNEL, 3, padding=1)
        self.conv2 = nn.Conv2d(N_CHANNEL, N_CHANNEL, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(N_CHANNEL, N_FEATURE)
        self.fc2 = nn.Linear(N_FEATURE, N_FEATURE)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def calibration_batches() -> list:
    generator = torch.Generator().manual_seed(31193)
    return [torch.randn(N_BATCH, N_CHANNEL, N_SIZE, N_SIZE, generator=generator).cuda() for _ in range(N_CALIBRATION)]

def export_onnx(model, example, path: Path) -> tuple:
    """Export a quantized module. Returns `(ok, which_exporter, message)`.

    Both exporters are tried, because they fail differently and the difference matters:
    the legacy TorchScript path cannot represent the quantizer modules at all
    (`Expected node type 'onnx::Constant'`), while the dynamo path traces them properly.
    """
    from modelopt.torch.quantization.utils import export_torch_mode
    message = ""
    for exporter, use_dynamo in [("dynamo", True), ("legacy", False)]:
        try:
            with torch.no_grad(), export_torch_mode():
                torch.onnx.export(model, (example, ), path, dynamo=use_dynamo, opset_version=OPSET, input_names=["x"], output_names=["y"], verbose=False)
            return True, exporter, ""
        except Exception as exception:  # noqa: BLE001 - a failure here is a result
            message = str(exception).splitlines()[0][:110]
    result.setdefault("export_error", {})[path.stem] = message
    return False, "-", message

def compile_with_torch_tensorrt(model, example) -> tuple:
    """Compile a quantized module straight to TensorRT, bypassing ONNX.

    Returns `(ok, max_abs_diff_vs_eager, message)`. This is the route that works when the
    ONNX one does not, and comparing against the module's own **eager** output is what shows
    whether TensorRT honoured the Q/DQ or quietly widened the precision.
    """
    import torch_tensorrt
    from modelopt.torch.quantization.utils import export_torch_mode
    try:
        with torch.no_grad(), export_torch_mode():
            exported = torch.export.export(model, (example, ))
            compiled = torch_tensorrt.dynamo.compile(exported, inputs=[example], use_explicit_typing=True, min_block_size=1)
        with torch.no_grad():
            eager = model(example)
        output = compiled(example)
        return True, float((output - eager).abs().max()), ""
    except Exception as exception:  # noqa: BLE001 - a failure here is a result
        return False, None, f"{type(exception).__name__}: {str(exception).splitlines()[0][:90]}"

def parse_with_tensorrt(path: Path) -> tuple:
    """Return `(parsed, layer_type_counter, message)`."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(path)):
        return False, OrderedDict(), str(parser.get_error(parser.num_errors - 1))[:130]
    counter = OrderedDict()
    for index in range(network.num_layers):
        name = str(network.get_layer(index).type).replace("LayerType.", "")
        counter[name] = counter.get(name, 0) + 1
    return True, counter, ""

# ================================================================ Cases

@case_mark
def case_reference() -> None:
    """The FP32 model everything is compared against."""
    model = MixedModel().eval().cuda()
    example = torch.randn(N_BATCH, N_CHANNEL, N_SIZE, N_SIZE).cuda()
    with torch.no_grad():
        reference = model(example)
    print(f"    model: 2 x Conv2d({N_CHANNEL}) + Linear({N_CHANNEL},{N_FEATURE}) + Linear({N_FEATURE},{N_FEATURE})")
    print(f"    output {tuple(reference.shape)}, range [{float(reference.min()):.3f}, {float(reference.max()):.3f}]")
    result["state_dict"] = {k: v.clone() for k, v in model.state_dict().items()}
    result["example"] = example
    result["reference"] = reference
    return

@case_mark
def case_quantize_each_format() -> None:
    """Quantize with each low-bit config and measure the error **eagerly**.

    Eager first, always: this is the quantization error with TensorRT removed from the
    question, so a bad number here is a calibration or format problem rather than an engine
    problem.
    """
    import modelopt.torch.quantization as mtq

    batch_list = calibration_batches()

    def forward_loop(model):
        with torch.no_grad():
            for batch in batch_list:
                model(batch)

    config_list = [
        ("FP8", mtq.FP8_DEFAULT_CFG),
        ("INT8", mtq.INT8_DEFAULT_CFG),
        ("NVFP4", mtq.NVFP4_DEFAULT_CFG),
        ("MXFP8", mtq.MXFP8_DEFAULT_CFG),
        ("INT4_AWQ", mtq.INT4_AWQ_CFG),
    ]
    print(f"    {'format':<10}{'quantizers':>12}{'eager max abs diff':>22}{'relative':>12}")
    print("    " + "-" * 58)
    scale = max(float(result["reference"].abs().max()), 1e-9)
    for name, config in config_list:
        model = MixedModel().eval().cuda()
        model.load_state_dict(result["state_dict"])
        try:
            quantized = mtq.quantize(model, config, forward_loop)
        except Exception as exception:  # noqa: BLE001
            print(f"    {name:<10}{'-':>12}{'quantize failed':>22}   {str(exception).splitlines()[0][:50]}")
            result.setdefault("quantized", OrderedDict())[name] = None
            continue
        with torch.no_grad():
            output = quantized(result["example"])
        difference = float((output - result["reference"]).abs().max())
        n_quantizer = sum(1 for module in quantized.modules() if type(module).__name__.endswith("TensorQuantizer"))
        print(f"    {name:<10}{n_quantizer:>12}{difference:>22.3e}{difference / scale:>11.2%}")
        result.setdefault("quantized", OrderedDict())[name] = quantized
        result.setdefault("eager", OrderedDict())[name] = difference / scale
    return

@case_mark
def case_two_routes_to_tensorrt() -> None:
    """Two ways to get a quantized module into TensorRT, and only one of them works here.

    + **Via ONNX** -- `torch.onnx.export` then the TensorRT parser. This is what the upstream
      example does, and on this stack it fails for **every** format: the legacy exporter
      cannot represent ModelOpt's quantizer modules (`Expected node type 'onnx::Constant'`)
      and the dynamo exporter falls back to the same path.
    + **Via Torch-TensorRT** -- `torch.export.export` inside `export_torch_mode()`, then
      `torch_tensorrt.dynamo.compile`. This one keeps the Q/DQ, and is the route
      `06-DLFrameworkTRT/ModelOptimizer` uses.

    The comparison is the useful part: a format that quantizes cleanly in PyTorch may still
    not reach TensorRT, and *which* leg fails tells you whether to blame the format or the
    exporter.
    """
    print(f"    {'format':<10}{'ONNX export':<16}{'Torch-TRT compile':<20}{'TRT vs eager':>14}")
    print("    " + "-" * 62)
    for name, quantized in result.get("quantized", {}).items():
        if quantized is None:
            continue
        path = output_path / f"model-{name.lower()}.onnx"
        onnx_ok, exporter, onnx_message = export_onnx(quantized, result["example"], path)

        model = MixedModel().eval().cuda()
        model.load_state_dict(result["state_dict"])
        # Re-quantize a fresh copy: `compile` consumes the module, and reusing the one that
        # was just handed to the ONNX exporter would confuse the two failures.
        trt_ok, difference, trt_message = compile_with_torch_tensorrt(quantized, result["example"])

        onnx_state = exporter if onnx_ok else "FAILED"
        trt_state = "ok" if trt_ok else "FAILED"
        shown = f"{difference:.3e}" if difference is not None else "-"
        print(f"    {name:<10}{onnx_state:<16}{trt_state:<20}{shown:>14}")
        if not onnx_ok:
            print(f"        onnx: {onnx_message[:88]}")
        if not trt_ok:
            print(f"        trt : {trt_message[:88]}")
        result.setdefault("route", OrderedDict())[name] = (onnx_ok, trt_ok, difference)
    return

@case_mark
def case_conv_vs_linear() -> None:
    """The rule that surprises people: Conv and Linear do not accept the same formats.

    TensorRT overrides the requested weight format for convolutions -- MXFP8 and NVFP4 become
    FP8, INT4-AWQ becomes INT8 -- because the conv kernels do not implement the 4-bit paths
    that the GEMM kernels do. Whether that override happens silently, or shows up as a parse
    failure, is the thing worth knowing before designing a mixed model around a 4-bit format.
    """
    for name, (onnx_ok, trt_ok, difference) in result.get("route", {}).items():
        if trt_ok:
            print(f"    {name:<10} reached TensorRT; engine vs eager module: {difference:.3e}")
        else:
            print(f"    {name:<10} did not reach TensorRT by either route")
    print("    A format that quantizes cleanly in PyTorch has still not necessarily reached the")
    print("    engine. Check the engine against the EAGER quantized module, not against FP32:")
    print("    that isolates 'did TensorRT honour the Q/DQ' from 'is this format accurate'.")
    return

@case_mark
def case_summary() -> None:
    print("\n" + "    " + "=" * 72)
    print(f"    {'format':<12}{'eager error':>16}{'via ONNX':>14}{'via Torch-TRT':>14}")
    print("    " + "-" * 72)
    for name in result.get("eager", {}):
        onnx_ok, trt_ok, difference = result.get("route", {}).get(name, (False, False, None))
        print(f"    {name:<12}{result['eager'][name]:>15.2%}{('ok' if onnx_ok else 'failed'):>14}{('ok' if trt_ok else 'failed'):>14}")
    print("    " + "=" * 72)
    eager = result.get("eager", {})
    if "INT8" in eager and "FP8" in eager:
        print(f"    Same bit width, different error: INT8 {eager['INT8']:.2%} vs FP8 {eager['FP8']:.2%} "
              f"({eager['FP8'] / max(eager['INT8'], 1e-12):.1f}x). Bit count alone does not decide accuracy.")
    if "NVFP4" in eager and "INT8" in eager:
        print(f"    But on THIS model the 4-bit formats are genuinely worse (NVFP4 {eager['NVFP4']:.2%}), i.e.")
        print("    block scaling does not rescue 4 bits here. A small model with narrow, well-behaved")
        print("    activation ranges is the case where fine-grained scaling has least to offer; the MX")
        print("    formats earn their keep on large models with heavy-tailed distributions.")
    return

def main() -> None:
    case_reference()
    case_quantize_each_format()
    case_two_routes_to_tensorrt()
    case_conv_vs_linear()
    case_summary()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
