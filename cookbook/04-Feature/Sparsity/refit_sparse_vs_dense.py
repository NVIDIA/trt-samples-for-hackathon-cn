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
"""Refitting an engine that was built with sparse weights: what may the new weights be?

`main.py` shows the `SPARSE_WEIGHTS` builder flag. This file asks the question that only
comes up when sparsity meets refit, and it is a question with a real answer:

**If the builder chose a 2:4-sparse kernel because the original weights were 2:4 sparse,
can you later refit that engine with dense weights?**

The kernel is baked into the plan. A 2:4-sparse kernel reads two values per group of four
and a metadata index that says which two -- so handing it a dense tensor is not a matter of
copying more bytes, the kernel physically cannot use them. The interesting part is what
TensorRT does about it: refuse, silently drop the extra values, or something else.

This is the rare case where refit has a *semantic* constraint rather than a shape one, which
is why it is worth its own file. `04-Feature/Refit` covers the ordinary API.

2:4 structured sparsity means: in every group of 4 consecutive values along the input-channel
axis, at most 2 are non-zero. `make_sparse_weight` below produces exactly that pattern.

Re-expressed from the idea in the internal `tutApiSparseConvTests.cpp`; no code was taken.
"""

import json
from collections import OrderedDict

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import case_mark

np.random.seed(31193)

N_BATCH = 1
N_CHANNEL_IN = 32
N_CHANNEL_OUT = 32
N_SIZE = 16
KERNEL = 3

result = OrderedDict()

def make_sparse_weight(seed: int = 0) -> np.ndarray:
    """A 2:4-structured-sparse convolution kernel, shape [out, in, kh, kw].

    The 2:4 pattern is along the **input-channel** axis, which is the axis a convolution
    reduces over and the one the sparse tensor cores index.
    """
    rng = np.random.default_rng(31193 + seed)
    weight = rng.standard_normal((N_CHANNEL_OUT, N_CHANNEL_IN, KERNEL, KERNEL)).astype(np.float32) * 0.05
    for group_start in range(0, N_CHANNEL_IN, 4):
        # Keep 2 of every 4, zero the other 2
        for out_index in range(N_CHANNEL_OUT):
            keep = rng.choice(4, size=2, replace=False)
            for offset in range(4):
                if offset not in keep:
                    weight[out_index, group_start + offset] = 0.0
    return np.ascontiguousarray(weight)

def make_dense_weight(seed: int = 0) -> np.ndarray:
    """The same shape with no zero structure at all."""
    rng = np.random.default_rng(1000 + seed)
    return np.ascontiguousarray(rng.standard_normal((N_CHANNEL_OUT, N_CHANNEL_IN, KERNEL, KERNEL)).astype(np.float32) * 0.05)

def sparsity_ratio(weight: np.ndarray) -> float:
    return float(np.count_nonzero(weight == 0.0) / weight.size)

def is_two_four(weight: np.ndarray) -> bool:
    """True when every group of 4 along the input-channel axis has at most 2 non-zeros."""
    reshaped = weight.reshape(weight.shape[0], weight.shape[1] // 4, 4, *weight.shape[2:])
    return bool(np.all(np.count_nonzero(reshaped, axis=2) <= 2))

def build_engine(weight: np.ndarray, *, b_sparse_flag: bool, b_refit: bool):
    """One convolution, optionally built for sparsity and/or refit.

    Returns `(engine, plan_size, tactic_list)`. The tactic names are the direct evidence of
    whether a sparse kernel was chosen -- plan size is not, since the flag itself changes it.
    """
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    if b_sparse_flag:
        builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    if b_refit:
        builder_config.set_flag(trt.BuilderFlag.REFIT)

    tensor = network.add_input("input", trt.float32, [N_BATCH, N_CHANNEL_IN, N_SIZE, N_SIZE])
    layer = network.add_convolution_nd(tensor, N_CHANNEL_OUT, [KERNEL, KERNEL], trt.Weights(weight), trt.Weights(np.zeros(N_CHANNEL_OUT, dtype=np.float32)))
    layer.padding_nd = [1, 1]
    layer.name = "conv"
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)

    builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    engine_bytes = builder.build_serialized_network(network, builder_config)
    assert engine_bytes is not None, "Failed building engine"
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)
    information = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    tactic_list = [layer.get("TacticName", "") for layer in information.get("Layers", []) if isinstance(layer, dict)]
    return engine, len(bytes(engine_bytes)), tactic_list

def run(engine, input_data: np.ndarray) -> np.ndarray:
    """Run once and return the output."""
    import cuda.bindings.runtime as cudart
    context = engine.create_execution_context()
    buffer = OrderedDict()
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = context.get_tensor_shape(name)
        n_byte = trt.volume(shape) * engine.get_tensor_dtype(name).itemsize
        buffer[name] = (cudart.cudaMalloc(n_byte)[1], n_byte, tuple(shape))
        context.set_tensor_address(name, buffer[name][0])
    cudart.cudaMemcpy(buffer["input"][0], input_data.ctypes.data, buffer["input"][1], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_async_v3(0)
    cudart.cudaStreamSynchronize(0)
    output = np.empty(buffer["output"][2], dtype=np.float32)
    cudart.cudaMemcpy(output.ctypes.data, buffer["output"][0], buffer["output"][1], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    for address, _, _ in buffer.values():
        cudart.cudaFree(address)
    return output

def refit_with(engine, weight: np.ndarray) -> tuple:
    """Try to refit `conv`'s kernel. Returns `(succeeded, message)`."""
    message_list = []

    class RecordingLogger(trt.ILogger):

        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, message):
            if severity <= trt.ILogger.Severity.ERROR:
                message_list.append(message)

    refitter = trt.Refitter(engine, RecordingLogger())
    try:
        ok = refitter.set_weights("conv", trt.WeightsRole.KERNEL, trt.Weights(weight))
        if ok:
            ok = refitter.refit_cuda_engine()
    except Exception as exception:  # noqa: BLE001 - the message is the result
        return False, str(exception).splitlines()[0]
    return bool(ok), (message_list[0] if message_list else "")

# ================================================================ Cases

@case_mark
def case_weights_are_really_sparse() -> None:
    """Confirm the test data really is 2:4 sparse before drawing conclusions from it."""
    sparse = make_sparse_weight()
    dense = make_dense_weight()
    print(f"    sparse weight: zeros {sparsity_ratio(sparse):.1%}, 2:4 structured = {is_two_four(sparse)}")
    print(f"    dense  weight: zeros {sparsity_ratio(dense):.1%}, 2:4 structured = {is_two_four(dense)}")
    assert is_two_four(sparse) and not is_two_four(dense), "The two weight sets must differ in structure"
    result["sparse"], result["dense"] = sparse, dense
    return

@case_mark
def case_build_both() -> None:
    """Build with and without `SPARSE_WEIGHTS`, and see whether the plan differs."""
    tactic_by_name = {}
    for name, weight, flag in [("sparse weights, flag on", result["sparse"], True), ("sparse weights, flag off", result["sparse"], False), ("dense weights, flag on", result["dense"], True)]:
        engine, n_byte, tactic_list = build_engine(weight, b_sparse_flag=flag, b_refit=False)
        compute = [t for t in tactic_list if t and "Move" not in t]
        tactic_by_name[name] = compute
        print(f"    {name:<26} plan {n_byte:>8} B   tactic {compute[0][:52] if compute else '(none)'}")

    on = tactic_by_name["sparse weights, flag on"]
    off = tactic_by_name["sparse weights, flag off"]
    same = on == off
    print(f"    tactic identical with and without SPARSE_WEIGHTS: {same}")
    if same:
        print("    -> the builder DECLINED sparsity for this layer. The flag is permission, not")
        print("       instruction: a sparse kernel is used only when allowed AND faster, and on")
        print("       this shape/arch it was not chosen. Everything below is therefore a property")
        print("       of a dense engine, and says nothing about a genuinely sparse one.")
    result["sparse_kernel_selected"] = not same
    return

@case_mark
def case_refit_sparse_with_sparse() -> None:
    """The supported case: refit a sparse-built engine with new **sparse** weights."""
    engine, _, _ = build_engine(result["sparse"], b_sparse_flag=True, b_refit=True)
    input_data = np.ascontiguousarray(np.random.rand(N_BATCH, N_CHANNEL_IN, N_SIZE, N_SIZE).astype(np.float32))
    before = run(engine, input_data)

    new_sparse = make_sparse_weight(seed=7)
    ok, message = refit_with(engine, new_sparse)
    print(f"    refit with 2:4-sparse weights: {ok}   {message[:100]}")
    assert ok, f"Refitting sparse with sparse should work: {message}"

    after = run(engine, input_data)
    changed = float(np.max(np.abs(after - before)))
    print(f"    output changed after refit: max |diff| = {changed:.3e}")
    assert changed > 0, "The refit did not take effect"
    result["sparse_refit"] = True
    return

@case_mark
def case_refit_sparse_with_dense() -> None:
    """The question this file exists for: refit that same engine with **dense** weights.

    The plan may hold a 2:4-sparse kernel that physically cannot read a dense tensor. So one
    of three things happens, and which one it is decides whether a deployment can hot-swap
    weights freely:

    + refused at `set_weights` / `refit_cuda_engine` -- safe, and the caller knows;
    + accepted and the extra values silently ignored -- the dangerous outcome, a wrong model
      that still runs;
    + accepted and correct, because the builder did not actually choose a sparse kernel.
    """
    engine, _, _ = build_engine(result["sparse"], b_sparse_flag=True, b_refit=True)
    input_data = np.ascontiguousarray(np.random.rand(N_BATCH, N_CHANNEL_IN, N_SIZE, N_SIZE).astype(np.float32))

    dense = result["dense"]
    ok, message = refit_with(engine, dense)
    print(f"    refit with dense weights: accepted = {ok}   {message[:100]}")

    if ok:
        # Accepted. Now the important part: are the results the ones dense weights imply?
        output = run(engine, input_data)
        reference = build_engine(dense, b_sparse_flag=False, b_refit=False)[0]
        expected = run(reference, input_data)
        difference = float(np.max(np.abs(output - expected)))
        print(f"    refitted output vs a freshly built dense engine: max |diff| = {difference:.3e}")
        if difference < 1e-3:
            print("    -> the dense weights were used in full; this engine did not hold a sparse kernel,")
            print("       so refit is unconstrained here.")
        else:
            print("    -> ACCEPTED BUT WRONG: the engine kept a sparse kernel and ignored the extra")
            print("       values. This is the silent failure to watch for.")
        result["dense_refit"] = ("accepted", difference)
    else:
        print("    -> refused, which is the safe outcome: the caller is told rather than misled.")
        result["dense_refit"] = ("refused", None)
    return

@case_mark
def case_summary() -> None:
    """State the rule this machine actually demonstrates."""
    print(f"    refit sparse-built engine with sparse weights: {'OK' if result.get('sparse_refit') else 'failed'}")
    status, difference = result["dense_refit"]
    print(f"    refit sparse-built engine with dense  weights: {status}" + (f", max |diff| vs dense reference = {difference:.3e}" if difference is not None else ""))
    print("    Note `SPARSE_WEIGHTS` is permission rather than instruction; whether a sparse kernel")
    print("    was actually selected depends on the shapes, the arch and the tactic timing, so this")
    print("    result is a property of this engine, not a universal rule. Re-run the check on yours.")
    return

def main() -> None:
    case_weights_are_really_sparse()
    case_build_both()
    case_refit_sparse_with_sparse()
    case_refit_sparse_with_dense()
    case_summary()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
