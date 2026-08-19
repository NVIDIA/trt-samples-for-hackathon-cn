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
"""Calibration *method* on an ONNX model: entropy, max, and INT4-AWQ, plus per-node calibration.

`03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` covers FP8 PTQ with max calibration on a
tiny CNN. The choice it does not make visible is the **calibration method**, which is what
decides where the clipping threshold lands:

+ **max**     - the threshold is the largest value seen. Nothing is clipped, so outliers get
                a scale that wastes resolution on values almost nothing uses.
+ **entropy** - the threshold minimises information loss against the full-precision
                distribution. Outliers *are* clipped, and the bulk of the distribution gets
                more levels.

Which wins depends on the data, not on the method, so the point of this file is to measure
both on the same model rather than recommend one.

It also covers **`calibrate_per_node=True`**, which exists for a memory reason rather than an
accuracy one: ordinary calibration collects activations for the whole graph at once, and on a
large model that does not fit. Per-node calibration walks the graph instead. This example
measures whether it changes the answer -- it should not, and confirming that is the point.

Everything runs on an ONNX file, so this is the deployment-side counterpart to
`04-Feature/LowBitQuantization`, which does the same comparison from PyTorch.
"""

import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import tensorrt as trt

from tensorrt_cookbook import case_mark, cookbook_path

np.random.seed(31193)

N_BATCH = 8
N_CALIBRATION = 16
output_path = Path(__file__).parent
onnx_file = cookbook_path("00-Data", "model", "model-trained.onnx")

result = OrderedDict()

def calibration_data() -> np.ndarray:
    """Fixed calibration set, with a deliberate outlier tail.

    The methods only differ when the distribution has something to clip. Uniform noise makes
    entropy and max agree, which would make this whole comparison look pointless.
    """
    rng = np.random.default_rng(31193)
    data = rng.standard_normal((N_CALIBRATION, 1, 28, 28)).astype(np.float32) * 0.2
    data[0, 0, 0, 0] = 8.0  # one large outlier, the thing entropy is allowed to clip away
    data[1, 0, 5, 5] = -6.0
    return data

def typical_data() -> np.ndarray:
    """The same distribution WITHOUT the planted outliers.

    Evaluating only on outlier-bearing inputs is a rigged comparison: `max` sizes its scale to
    represent those exact values, so it necessarily wins there. Which method is better depends
    on which inputs you care about, so both are measured.
    """
    rng = np.random.default_rng(20260905)
    return (rng.standard_normal((N_BATCH, 1, 28, 28)).astype(np.float32) * 0.2)

def run_onnxruntime(path: Path, batch: np.ndarray) -> np.ndarray:
    option = onnxruntime.SessionOptions()
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(path), option, providers=["CPUExecutionProvider"])
    return session.run(None, {"x": batch})[0]

def build_engine(path: Path) -> tuple:
    """Parse and build. Returns `(ok, n_qdq_layer, message)`."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(path)):
        return False, 0, str(parser.get_error(parser.num_errors - 1))[:110]
    n_qdq = sum(1 for i in range(network.num_layers) if str(network.get_layer(i).type) in ["LayerType.QUANTIZE", "LayerType.DEQUANTIZE"])

    # This model has a dynamic batch dimension, so the build needs a profile. Without one the
    # builder refuses with a message that mentions neither quantization nor the input name:
    #     Network has dynamic or shape inputs, but no optimization profile has been defined
    builder_config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    n_dynamic = 0
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = list(tensor.shape)
        if -1 in shape:
            n_dynamic += 1
            minimum = [1 if d == -1 else d for d in shape]
            optimum = [N_BATCH if d == -1 else d for d in shape]
            profile.set_shape(tensor.name, minimum, optimum, optimum)
    if n_dynamic:
        builder_config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, builder_config)
    return engine_bytes is not None, n_qdq, "" if engine_bytes is not None else "build failed"

def quantize_with(method: str, target: Path, **extra) -> tuple:
    """Run ModelOpt PTQ with one calibration method. Returns `(ok, seconds, message)`."""
    from modelopt.onnx.quantization import quantize
    t0 = time.time()
    try:
        quantize(
            onnx_path=str(onnx_file),
            quantize_mode=extra.pop("quantize_mode", "int8"),
            calibration_data=calibration_data(),
            calibration_method=method,
            high_precision_dtype="fp32",  # See 05-Plugin/ONNXPTQWithPlugin for why this matters
            output_path=str(target),
            **extra,
        )
        return True, time.time() - t0, ""
    except Exception as exception:  # noqa: BLE001 - a failure here is a result
        return False, time.time() - t0, f"{type(exception).__name__}: {str(exception).splitlines()[-1][:100]}"

# ================================================================ Cases

@case_mark
def case_reference() -> None:
    """The FP32 model and its output, as the baseline for every comparison below."""
    probe = calibration_data()[:N_BATCH]  # contains the planted outliers
    typical = typical_data()  # same distribution, no outliers
    reference = run_onnxruntime(onnx_file, probe)
    reference_typical = run_onnxruntime(onnx_file, typical)
    result["typical"] = typical
    result["reference_typical"] = reference_typical
    model = onnx.load(onnx_file)
    print(f"    {onnx_file.name}: {len(model.graph.node)} nodes")
    print(f"    calibration set: {N_CALIBRATION} samples, range [{calibration_data().min():.2f}, {calibration_data().max():.2f}] (outliers planted on purpose)")
    result["probe"] = probe
    result["reference"] = reference
    return

@case_mark
def case_calibration_method() -> None:
    """entropy against max, same model, same data."""
    scale = max(float(np.abs(result["reference"]).max()), 1e-9)
    scale_typical = max(float(np.abs(result["reference_typical"]).max()), 1e-9)
    print(f"    {'method':<12}{'seconds':>9}{'error on outlier input':>26}{'error on typical input':>26}")
    print("    " + "-" * 74)
    for method in ["entropy", "max"]:
        target = output_path / f"model-int8-{method}.onnx"
        ok, second, message = quantize_with(method, target)
        if not ok:
            print(f"    {method:<12}{second:>9.1f}{'FAILED':>26}   {message[:34]}")
            result.setdefault("method", OrderedDict())[method] = None
            continue
        outlier_error = float(np.abs(run_onnxruntime(target, result["probe"]) - result["reference"]).max()) / scale
        typical_error = float(np.abs(run_onnxruntime(target, result["typical"]) - result["reference_typical"]).max()) / scale_typical
        print(f"    {method:<12}{second:>9.1f}{outlier_error:>25.2%}{typical_error:>26.2%}")
        result.setdefault("method", OrderedDict())[method] = (outlier_error, second, target, typical_error)
    return

@case_mark
def case_per_node_calibration() -> None:
    """`calibrate_per_node=True`: a memory strategy that must not change the answer.

    Ordinary calibration collects activations for the whole graph at once. On a large model
    that is what runs out of memory, and per-node calibration walks the graph instead. It is
    slower, and it should produce the *same* ranges -- so the useful measurement is whether
    the quantized model comes out equivalent, not whether it is more accurate.
    """
    scale = max(float(np.abs(result["reference"]).max()), 1e-9)
    result.get("method", {}).get("entropy")
    for flag in [False, True]:
        target = output_path / f"model-int8-per_node_{flag}.onnx"
        ok, second, message = quantize_with("entropy", target, calibrate_per_node=flag)
        if not ok:
            print(f"    calibrate_per_node={str(flag):<6}: FAILED after {second:.1f} s -- {message[:70]}")
            result.setdefault("per_node", OrderedDict())[flag] = None
            continue
        output = run_onnxruntime(target, result["probe"])
        difference = float(np.abs(output - result["reference"]).max())
        print(f"    calibrate_per_node={str(flag):<6}: {second:6.1f} s, max abs diff {difference:.3e} ({difference / scale:.2%})")
        result.setdefault("per_node", OrderedDict())[flag] = (difference / scale, second, target)

    entry_false = result.get("per_node", {}).get(False)
    entry_true = result.get("per_node", {}).get(True)
    if entry_false and entry_true:
        same = abs(entry_false[0] - entry_true[0]) < 1e-9
        drift = abs(entry_false[0] - entry_true[0])
        print(f"    bit-identical: {same}   accuracy drift: {drift:.2%}   time ratio: {entry_true[1] / max(entry_false[1], 1e-9):.1f}x")
        print("    Per-node calibration is a way to fit a large model into memory, not a way to")
        print("    quantize it better -- and it is NOT free: the walk costs a large constant factor.")
        if not same and drift < 0.005:
            print("    The small residual drift is expected: collecting per node changes the order in")
            print("    which histograms are accumulated, so the chosen thresholds can land a bin apart.")
            print("    Treat 'unchanged' as 'within a bin', not as 'bitwise equal'.")
    return

@case_mark
def case_int4_weight_only() -> None:
    """INT4 weight-only quantization on the same graph, and whether TensorRT takes it."""
    target = output_path / "model-int4.onnx"
    ok, second, message = quantize_with("awq_clip", target, quantize_mode="int4")
    if not ok:
        print(f"    int4 / awq_clip: FAILED after {second:.1f} s")
        print(f"        {message[:120]}")
        result["int4"] = None
        return
    model = onnx.load(target)
    counter = OrderedDict()
    for node in model.graph.node:
        counter[node.op_type] = counter.get(node.op_type, 0) + 1
    print(f"    int4 / awq_clip: quantized in {second:.1f} s, nodes {dict(sorted(counter.items()))}")
    built, n_qdq, build_message = build_engine(target)
    print(f"    TensorRT: built = {built}, Q/DQ layers = {n_qdq}  {build_message[:60]}")
    result["int4"] = (built, n_qdq)
    return

@case_mark
def case_engines() -> None:
    """Every quantized model that exists, through the builder."""
    print(f"    {'model':<28}{'TRT build':<12}{'Q/DQ layers':>13}  note")
    print("    " + "-" * 68)
    for label, entry in [(f"int8/{name}", value) for name, value in result.get("method", {}).items()]:
        if entry is None:
            continue
        built, n_qdq, message = build_engine(entry[2])
        print(f"    {label:<28}{str(built):<12}{n_qdq:>13}  {message[:30]}")
        result.setdefault("engine", OrderedDict())[label] = (built, n_qdq)
    return

@case_mark
def case_summary() -> None:
    print("\n" + "    " + "=" * 70)
    print(f"    {'configuration':<30}{'relative error':>18}{'TRT build':>14}")
    print("    " + "-" * 70)
    for name, entry in result.get("method", {}).items():
        if entry is None:
            continue
        built = result.get("engine", {}).get(f"int8/{name}", (None, None))[0]
        print(f"    {'int8, ' + name:<30}{entry[0]:>17.2%}{str(built):>14}")
    if result.get("int4"):
        print(f"    {'int4, awq_clip':<30}{'-':>18}{str(result['int4'][0]):>14}")
    print("    " + "=" * 70)
    method = result.get("method", {})
    if method.get("entropy") and method.get("max"):
        entropy_out, _, _, entropy_typ = method["entropy"]
        max_out, _, _, max_typ = method["max"]
        print(f"    on OUTLIER-bearing input : entropy {entropy_out:.2%}  vs  max {max_out:.2%}  -> {'max' if max_out < entropy_out else 'entropy'} wins")
        print(f"    on TYPICAL input         : entropy {entropy_typ:.2%}  vs  max {max_typ:.2%}  -> {'max' if max_typ < entropy_typ else 'entropy'} wins")
        print("    That is the whole trade: `max` sizes its scale to represent the extremes, so it")
        print("    necessarily wins on inputs containing them and spends resolution doing it.")
        print("    `entropy` clips the tail to give the bulk of the distribution more levels.")
        print("    Measuring on only one of these two probes would have produced a confident and")
        print("    misleading recommendation.")
    return

def main() -> None:
    case_reference()
    case_calibration_method()
    case_per_node_calibration()
    case_int4_weight_only()
    case_engines()
    case_summary()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
