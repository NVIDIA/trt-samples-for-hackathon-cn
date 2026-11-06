# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
"""Quantize / Dequantize layers, at all three granularities.

    per-tensor    1 scale for everything          coarsest, cheapest
    per-channel   1 scale per output channel      `axis`
    block         1 scale per block_shape region  `block_shape`, finest

The block cases were a separate `block_quantization.py` until they were merged here, so that
one file holds all three granularities and `check_api_coverage` below reports against all of
them at once.
"""

from collections import OrderedDict

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark, check_api_coverage, datatype_cast, print_enumerated_members

np.random.seed(31193)

N_M, N_K, N_N = 16, 64, 32
BLOCK = 32

result = OrderedDict()

def build_block_quantization(block_shape=None, to_type=trt.DataType.FP8, *, b_standalone: bool = False, scale_type=None):
    """Quantize a weight constant and feed it to a MatMul. Returns `(b_ok, message)`.

    Built with a raw builder rather than `TRTWrapperV1` on purpose: several cases below are
    *negative*, and the point of them is the error message TensorRT emits, which needs a logger
    that keeps it rather than one that prints and forgets.
    """
    message_list = []

    class RecordingLogger(trt.ILogger):

        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, message):
            if severity <= trt.ILogger.Severity.ERROR:
                message_list.append(message)

    logger = RecordingLogger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()

    x = network.add_input("x", trt.float16, [N_M, N_K])
    weight = np.ascontiguousarray((np.random.rand(N_K, N_N) * 0.1).astype(np.float16))
    w = network.add_constant([N_K, N_N], weight).get_output(0)

    if block_shape is None:  # per-tensor, for contrast
        scale_shape = []
    else:
        scale_shape = [N_K // block_shape[0], N_N // block_shape[1]]
    scale_value = np.ascontiguousarray(np.full(scale_shape if scale_shape else [], 0.02, dtype=np.float16))
    scale = network.add_constant(scale_shape, scale_value).get_output(0)

    if scale_type is not None:
        scale = network.add_cast(scale, scale_type).get_output(0)
    layer_q = network.add_quantize(w, scale, to_type)
    layer_dq = network.add_dequantize(layer_q.get_output(0), scale, trt.float16)
    if block_shape is not None:
        layer_q.block_shape = block_shape
        layer_q.axis = 0
        layer_dq.block_shape = block_shape
        layer_dq.axis = 0

    if b_standalone:
        output = layer_dq.get_output(0)
    else:
        output = network.add_matrix_multiply(x, trt.MatrixOperation.NONE, layer_dq.get_output(0), trt.MatrixOperation.NONE).get_output(0)
    output.name = "y"
    network.mark_output(output)

    engine_bytes = builder.build_serialized_network(network, builder_config)
    return engine_bytes is not None, (message_list[0] if message_list else "")

@case_mark
def case_simple():
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer_q_scale = tw.network.add_constant([], np.array([60 / 127], dtype=np.float32))
    layer_q_scale.shape = []  # Reset later
    layer_q_scale.weights = np.array([60 / 127], dtype=np.float32)  # Reset later
    layer_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer_dq_scale.shape = []  # Reset later
    layer_dq_scale.weights = np.array([1], dtype=np.float32)  # Reset later
    layer_q = tw.network.add_quantize(tensor, layer_q_scale.get_output(0), trt.DataType.INT8)
    # Input: input (T1: float16/bfloat16/float32), scale (T1, build-time constant, scalar/1D/block-rank), zero_point (optional T2: float32, must be all-zeros)
    # Outputs: output tensor of type T3 (int4, int8, float4, float8)
    # Data type: T1 (input/scale): float16, bfloat16, float32; T2 (zero_point): float32; T3 (output): int4, int8, float4, float8
    # Shape: input and output share shape [a0,...,an]; scale shape equals input shape with each dimension divided by corresponding block size
    # Volume limits: none specified
    layer_q.axis = 0  # [Optional] Default: -1 (must be set explicitly for per-channel quantization; negative axis raises error)
    layer_dq = tw.network.add_dequantize(layer_q.get_output(0), layer_dq_scale.get_output(0), trt.DataType.FLOAT)
    layer_dq.axis = 0  # [Optional] Default: -1 (must be set explicitly for per-channel dequantization; negative axis raises error)

    check_api_coverage(layer_q)  # Sanity check, unnecessary in normal workflow

    tw.build([layer_dq.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_axis():
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer_q_scale = tw.network.add_constant([4], np.array([40 / 127, 80 / 127, 120 / 127, 160 / 127], dtype=np.float32))
    layer_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer_q = tw.network.add_quantize(tensor, layer_q_scale.get_output(0), trt.DataType.INT8)
    layer_q.axis = 1
    layer_dq = tw.network.add_dequantize(layer_q.get_output(0), layer_dq_scale.get_output(0), trt.DataType.FLOAT)

    tw.build([layer_dq.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_set_input_zero_point():
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer_q_scale = tw.network.add_constant([4], np.array([20 / 127, 40 / 127, 60 / 127, 80 / 127], dtype=np.float32))
    layer_q_zeropoint = tw.network.add_constant([4], np.array([0, 0, 0, 0], dtype=np.float32))  # Only all-zeros is supported
    layer_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer_q = tw.network.add_quantize(tensor, layer_q_scale.get_output(0), trt.DataType.INT8)
    layer_q.axis = 1
    layer_q.set_input(2, layer_q_zeropoint.get_output(0))
    layer_dq = tw.network.add_dequantize(layer_q.get_output(0), layer_dq_scale.get_output(0), trt.DataType.FLOAT)
    layer_dq.axis = 0  # [Optional] Default: -1 (must be set explicitly for per-channel dequantization; negative axis raises error)

    tw.build([layer_dq.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_three_argument():
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer_q_scale = tw.network.add_constant([], np.array([60 / 127], dtype=np.float32))
    layer_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer_q = tw.network.add_quantize(tensor, layer_q_scale.get_output(0), trt.DataType.FP8)
    layer_dq = tw.network.add_dequantize(layer_q.get_output(0), layer_dq_scale.get_output(0), trt.DataType.FLOAT)

    tw.build([layer_dq.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_block_scale_shape() -> None:
    """The scale tensor's shape follows from `block_shape`; it is not a free choice."""
    for block_shape in [[32, 1], [16, 1], [64, 1], [32, 32]]:
        scale_shape = [N_K // block_shape[0], N_N // block_shape[1]]
        n_scale = int(np.prod(scale_shape))
        print(f"    weight [{N_K}, {N_N}]  block_shape={str(block_shape):<10} -> scale {str(scale_shape):<10} "
              f"({n_scale:>4} scales, {N_K * N_N // n_scale:>4} weights each)")
    print("    block_shape=[K, 1] with K = the whole axis is per-channel; [1, 1] would be per-element.")
    return

@case_mark
def case_block_into_matmul() -> None:
    """The supported shape: block-quantized weights feeding a MatMul."""
    b_ok, message = build_block_quantization(block_shape=[BLOCK, 1], to_type=trt.DataType.FP8)
    print(f"    FP8 block QDQ (block {BLOCK}) -> MatMul: built = {b_ok}   {message[:100]}")
    assert b_ok, f"Expected this to build: {message}"
    result["block_matmul"] = b_ok
    return

@case_mark
def case_block_fp4_is_rejected() -> None:
    """FP4 output is not allowed for `IQuantizeLayer` block quantization."""
    b_ok, message = build_block_quantization(block_shape=[BLOCK, 1], to_type=trt.DataType.FP4)
    print(f"    FP4 block QDQ -> MatMul: built = {b_ok}")
    print(f"        {message[:150]}")
    assert not b_ok, "FP4 block quantization was expected to be rejected here"
    print("    FP4 block quantization goes through IDynamicQuantizeLayer instead; see ../DynamicQuantize/.")
    result["fp4"] = "rejected"
    return

@case_mark
def case_block_scale_type() -> None:
    """Which scale type works, with and without a consumer -- a 2x2, because both were suspect.

    An earlier version of this code claimed a bare Q/DQ pair could not build. That was wrong:
    the failure came from the **scale type**, not from the missing consumer, and varying one
    thing at a time is what separated them.
    """
    print(f"    {'scale type':<12}{'standalone':<12}{'built':<8}message")
    for scale_type, label in [(None, "float16"), (trt.DataType.E8M0, "E8M0")]:
        for b_standalone in [False, True]:
            b_ok, message = build_block_quantization(block_shape=[BLOCK, 1], to_type=trt.DataType.FP8, b_standalone=b_standalone, scale_type=scale_type)
            print(f"    {label:<12}{str(b_standalone):<12}{str(b_ok):<8}{message[:60]}")
            result.setdefault("scale_type", []).append((label, b_standalone, b_ok))
    print("    -> the consumer is irrelevant; the E8M0 scale is what fails, in both shapes.")
    assert result["scale_type"][0][2] and result["scale_type"][1][2], "float16 scales should build"
    assert not result["scale_type"][2][2] and not result["scale_type"][3][2], "E8M0 scales were expected to fail here"
    return

@case_mark
def case_block_granularity_comparison() -> None:
    """Per-tensor against block, on the same weights, so the trade-off is visible."""
    for name, block_shape in [("per-tensor", None), ("block 32", [32, 1]), ("block 16", [16, 1])]:
        b_ok, message = build_block_quantization(block_shape=block_shape, to_type=trt.DataType.FP8)
        n_scale = 1 if block_shape is None else (N_K // block_shape[0]) * (N_N // block_shape[1])
        print(f"    {name:<12} scales={n_scale:>5}  built={b_ok}   {message[:70]}")
        result.setdefault("granularity", []).append((name, n_scale, b_ok))
    print("    More scales cost storage and a little arithmetic, and buy accuracy at low bit widths.")
    return

if __name__ == "__main__":
    # A simple QDQ structure
    case_simple()
    # Axis
    case_axis()
    # Use scale and zero point from earlier layer
    case_set_input_zero_point()
    # Three-argument quantization layer
    case_three_argument()
    # Block quantization: the third granularity, selected by `block_shape`
    case_block_scale_shape()
    case_block_into_matmul()
    case_block_fp4_is_rejected()
    case_block_scale_type()
    case_block_granularity_comparison()

    print_enumerated_members(trt.BuilderFlag)
    print_enumerated_members(trt.NetworkDefinitionCreationFlag)

    print("Finish")
