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
"""FP8 and INT8 post-training quantization through ModelOpt, compiled by the Dynamo frontend.

The path is: **PyTorch model -> `mtq.quantize` inserts Q/DQ -> `torch_tensorrt.dynamo.compile`
builds a strongly-typed engine.** Since TensorRT 11 removed weak-typing INT8 calibration,
this is the sanctioned way to get a quantized engine out of PyTorch, and the Q/DQ nodes
ModelOpt inserts are what tell the builder where the precision changes.

Three things worth knowing before reading the code:

+ **`mtq.quantize` needs a calibration loop, not just a config.** The `forward_loop` argument
  is called with the model; whatever it pushes through sets the amax values. A loop that
  feeds unrepresentative data produces a quantized model that is wrong in a way no shape
  check will find.
+ **The quantized module is still a PyTorch module.** It runs eagerly, so its accuracy can be
  checked *before* any TensorRT involvement -- which is the right place to find a calibration
  problem.
+ **`use_explicit_typing=True` is what makes the Q/DQ authoritative.** It tells Torch-TensorRT
  to honour the types in the graph instead of choosing precisions itself. Note that combining
  it with `enabled_precisions` is **accepted** in this version rather than rejected -- the
  documented error is not raised here, so the two settings can silently disagree. Measured in
  `case_strong_typing_and_enabled_precisions`.

Upstream (`quantize_vit_fp8.py`, `vgg16_ptq.py`) uses a timm ViT and CIFAR. This cookbook does
not download at run time, so the model here is a small CNN with the same structure -- conv
stack, pooling, classifier -- and the calibration set is synthetic but *fixed*, which is what
makes the numbers below reproducible.
"""

import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch_tensorrt

from tensorrt_cookbook import case_mark

np.random.seed(31193)
torch.manual_seed(31193)

N_BATCH = 32
N_CHANNEL = 3
N_SIZE = 32
N_CLASS = 10
N_CALIBRATION_BATCH = 8
N_WARMUP = 5
N_INFERENCE = 20

result = OrderedDict()

class SmallCNN(nn.Module):
    """Conv stack -> pool -> classifier: the shape of a vision model, small enough to build fast."""

    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(N_CHANNEL, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, N_CLASS)

    def forward(self, x):
        return self.classifier(self.feature(x).flatten(1))

def make_calibration_data() -> list:
    """A fixed synthetic calibration set.

    Deliberately *not* random every call: calibration determines the amax values, so a
    changing set makes every run produce a slightly different engine and an unreproducible
    accuracy number.
    """
    generator = torch.Generator().manual_seed(31193)
    return [torch.randn(N_BATCH, N_CHANNEL, N_SIZE, N_SIZE, generator=generator).cuda() for _ in range(N_CALIBRATION_BATCH)]

def measure(module, input_tensor) -> float:
    for _ in range(N_WARMUP):
        module(input_tensor)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_INFERENCE):
        module(input_tensor)
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000 / N_INFERENCE

def compile_dynamo(model, example, *, b_quantized: bool = False):
    """Compile with strong typing, which is what makes the Q/DQ authoritative.

    A ModelOpt-quantized module must be exported inside
    `modelopt.torch.quantization.utils.export_torch_mode()`. Without it,
    `torch.export.export` fails with

        RuntimeError: We found a fake tensor in the exported program constant's list

    because the quantizers hold fake tensors that only resolve in export mode. The message
    names neither ModelOpt nor quantization, which is what makes it hard to place.
    """
    if b_quantized:
        from modelopt.torch.quantization.utils import export_torch_mode
        with torch.no_grad(), export_torch_mode():
            exported = torch.export.export(model, (example, ))
    else:
        exported = torch.export.export(model, (example, ))
    return torch_tensorrt.dynamo.compile(exported, inputs=[example], use_explicit_typing=True, min_block_size=1)

# ================================================================ Cases

@case_mark
def case_baseline() -> None:
    """The unquantized model, in Torch and through TensorRT, as the reference."""
    model = SmallCNN().eval().cuda()
    example = torch.randn(N_BATCH, N_CHANNEL, N_SIZE, N_SIZE).cuda()

    with torch.no_grad():
        torch_output = model(example)
    compiled = compile_dynamo(model, example)
    trt_output = compiled(example)
    difference = float((trt_output - torch_output).abs().max())

    print(f"    FP32 Torch vs TensorRT: max |diff| = {difference:.3e}, latency {measure(compiled, example):.3f} ms")
    result["model"] = model
    result["example"] = example
    result["reference"] = torch_output.detach()
    result["fp32_latency"] = measure(compiled, example)
    return

@case_mark
def case_quantize() -> None:
    """`mtq.quantize` with a real calibration loop, for FP8 and INT8."""
    import modelopt.torch.quantization as mtq

    calibration_data = make_calibration_data()

    def forward_loop(model):
        """What ModelOpt calls to observe activation ranges. This is the calibration."""
        with torch.no_grad():
            for batch in calibration_data:
                model(batch)

    for name, config in [("FP8", mtq.FP8_DEFAULT_CFG), ("INT8", mtq.INT8_DEFAULT_CFG)]:
        model = SmallCNN().eval().cuda()
        model.load_state_dict(result["model"].state_dict())
        quantized = mtq.quantize(model, config, forward_loop)

        # Still a PyTorch module: check it eagerly before TensorRT is involved at all
        with torch.no_grad():
            eager_output = quantized(result["example"])
        eager_difference = float((eager_output - result["reference"]).abs().max())
        relative = eager_difference / max(float(result["reference"].abs().max()), 1e-9)
        n_quantizer = sum(1 for module in quantized.modules() if type(module).__name__.endswith("TensorQuantizer"))
        print(f"    {name}: {n_quantizer} quantizers inserted, eager max |diff| vs FP32 = {eager_difference:.3e} (relative {relative:.2%})")
        result.setdefault("quantized", OrderedDict())[name] = quantized
        result.setdefault("eager_difference", OrderedDict())[name] = (eager_difference, relative)
    return

@case_mark
def case_compile_quantized() -> None:
    """Compile each quantized model and confirm TensorRT honours the Q/DQ."""
    for name, quantized in result["quantized"].items():
        try:
            compiled = compile_dynamo(quantized, result["example"], b_quantized=True)
            output = compiled(result["example"])
            difference = float((output - result["reference"]).abs().max())
            latency = measure(compiled, result["example"])
            eager_difference = result["eager_difference"][name][0]
            print(f"    {name}: compiled OK, TRT max |diff| vs FP32 = {difference:.3e} "
                  f"(eager was {eager_difference:.3e}), latency {latency:.3f} ms")
            result.setdefault("compiled", OrderedDict())[name] = (difference, latency)
        except Exception as exception:  # noqa: BLE001 - a failure here is a result
            print(f"    {name}: compile FAILED -> {type(exception).__name__}: {str(exception).splitlines()[0][:120]}")
            result.setdefault("compiled", OrderedDict())[name] = None
    return

@case_mark
def case_strong_typing_and_enabled_precisions() -> None:
    """Do the two precision controls conflict? Measured, because the answer is version-specific.

    Coming from TensorRT 8/10 habits the instinct is to *ask* for a precision with
    `enabled_precisions`; the strongly-typed way is to put Q/DQ in the graph and let it speak.
    Documentation for other versions says passing both is an error.

    **In `torch_tensorrt` 2.14.0a0 it is not an error -- it is accepted silently.** That is
    worse than a rejection: the two settings can disagree and nothing says so, which is
    exactly the situation `04-Feature/BuilderFlag` warns about for the TensorRT flags.
    """
    model = result["model"]
    exported = torch.export.export(model, (result["example"], ))
    try:
        torch_tensorrt.dynamo.compile(exported, inputs=[result["example"]], use_explicit_typing=True, enabled_precisions={torch.float16}, min_block_size=1)
        print("    use_explicit_typing + enabled_precisions: ACCEPTED (no error raised)")
        print("    -> do not rely on being told; pass one or the other, not both")
        result["both_accepted"] = True
    except Exception as exception:  # noqa: BLE001 - the message is the lesson
        print(f"    use_explicit_typing + enabled_precisions -> {type(exception).__name__}")
        print(f"        {str(exception).splitlines()[0][:140]}")
        result["both_accepted"] = False
    return

@case_mark
def case_summary() -> None:
    print("\n" + "    " + "=" * 74)
    print(f"    {'precision':<12}{'eager |diff|':>16}{'TensorRT |diff|':>18}{'latency ms':>14}")
    print("    " + "-" * 74)
    print(f"    {'FP32':<12}{0.0:>16.3e}{0.0:>18.3e}{result['fp32_latency']:>14.3f}")
    for name in result.get("compiled", {}):
        entry = result["compiled"][name]
        eager = result["eager_difference"][name][0]
        if entry is None:
            print(f"    {name:<12}{eager:>16.3e}{'compile failed':>18}{'-':>14}")
        else:
            print(f"    {name:<12}{eager:>16.3e}{entry[0]:>18.3e}{entry[1]:>14.3f}")
    print("    " + "=" * 74)
    print("    The eager column is the one to read first: it is the quantization error alone,")
    print("    with TensorRT removed from the question.")
    return

def main() -> None:
    case_baseline()
    case_quantize()
    case_compile_quantized()
    case_strong_typing_and_enabled_precisions()
    case_summary()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
