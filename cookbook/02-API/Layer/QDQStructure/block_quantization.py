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
"""Block quantization: one scale per block of weights, not per tensor and not per channel.

`main.py` covers per-tensor and per-channel Q/DQ. **Block** quantization is the third
granularity and the one the MX formats are built on: the scale tensor has the same rank as
the data, and `block_shape` says how many elements share each scale.

    per-tensor    1 scale for everything          coarsest, cheapest
    per-channel   1 scale per output channel      `axis`
    block         1 scale per block_shape region  `block_shape`, finest

For a `[64, 32]` weight with `block_shape=[32, 1]`, the scale tensor is `[2, 32]`: every
column is split into two blocks of 32 rows, and each block gets its own scale. Finer scales
track the weight distribution better, which is what makes 4- and 8-bit weights usable.

Three constraints, all measured here rather than quoted:

1. **The output type must be INT8 or FP8-E4M3.** Asking for FP4 is rejected with
   `Blockwise quantization requires output type to be int8 or fp8e4m3`. FP4 block
   quantization exists, but through `IDynamicQuantizeLayer` -- see `../DynamicQuantize/`.
2. **An E8M0 scale does not build**, with or without a consumer. E8M0 is the shared-exponent
   scale type of the MX formats and `trt.DataType.E8M0` exists, but feeding one to
   `IQuantizeLayer` fails inside Myelin with an NVRTC compilation error rather than a usage
   message. A float16 scale of the same shape builds fine.
3. **The scale tensor's shape is derived, not free**: `data.shape[i] / block_shape[i]`.
"""

from collections import OrderedDict

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import case_mark

np.random.seed(31193)

N_M, N_K, N_N = 16, 64, 32
BLOCK = 32

result = OrderedDict()

def build(block_shape=None, to_type=trt.DataType.FP8, *, b_standalone: bool = False, scale_type=None):
    """Quantize a weight constant and feed it to a MatMul. Returns `(ok, message)`."""
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

# ================================================================ Cases

@case_mark
def case_scale_shape() -> None:
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
    ok, message = build(block_shape=[BLOCK, 1], to_type=trt.DataType.FP8)
    print(f"    FP8 block QDQ (block {BLOCK}) -> MatMul: built = {ok}   {message[:100]}")
    assert ok, f"Expected this to build: {message}"
    result["block_matmul"] = ok
    return

@case_mark
def case_fp4_is_rejected() -> None:
    """FP4 output is not allowed for `IQuantizeLayer` block quantization."""
    ok, message = build(block_shape=[BLOCK, 1], to_type=trt.DataType.FP4)
    print(f"    FP4 block QDQ -> MatMul: built = {ok}")
    print(f"        {message[:150]}")
    assert not ok, "FP4 block quantization was expected to be rejected here"
    print("    FP4 block quantization goes through IDynamicQuantizeLayer instead; see ../DynamicQuantize/.")
    result["fp4"] = "rejected"
    return

@case_mark
def case_scale_type() -> None:
    """Which scale type works, with and without a consumer -- a 2x2, because both were suspect.

    An earlier version of this file claimed a bare Q/DQ pair could not build. That was wrong:
    the failure came from the **scale type**, not from the missing consumer, and varying one
    thing at a time is what separated them.
    """
    print(f"    {'scale type':<12}{'standalone':<12}{'built':<8}message")
    for scale_type, label in [(None, "float16"), (trt.DataType.E8M0, "E8M0")]:
        for b_standalone in [False, True]:
            ok, message = build(block_shape=[BLOCK, 1], to_type=trt.DataType.FP8, b_standalone=b_standalone, scale_type=scale_type)
            print(f"    {label:<12}{str(b_standalone):<12}{str(ok):<8}{message[:60]}")
            result.setdefault("scale_type", []).append((label, b_standalone, ok))
    print("    -> the consumer is irrelevant; the E8M0 scale is what fails, in both shapes.")
    assert result["scale_type"][0][2] and result["scale_type"][1][2], "float16 scales should build"
    assert not result["scale_type"][2][2] and not result["scale_type"][3][2], "E8M0 scales were expected to fail here"
    return

@case_mark
def case_granularity_comparison() -> None:
    """Per-tensor against block, on the same weights, so the trade-off is visible."""
    for name, block_shape in [("per-tensor", None), ("block 32", [32, 1]), ("block 16", [16, 1])]:
        ok, message = build(block_shape=block_shape, to_type=trt.DataType.FP8)
        n_scale = 1 if block_shape is None else (N_K // block_shape[0]) * (N_N // block_shape[1])
        print(f"    {name:<12} scales={n_scale:>5}  built={ok}   {message[:70]}")
        result.setdefault("granularity", []).append((name, n_scale, ok))
    print("    More scales cost storage and a little arithmetic, and buy accuracy at low bit widths.")
    return

def main() -> None:
    case_scale_shape()
    case_block_into_matmul()
    case_fp4_is_rejected()
    case_scale_type()
    case_granularity_comparison()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
