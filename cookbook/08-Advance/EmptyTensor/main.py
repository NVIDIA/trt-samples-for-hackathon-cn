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
"""Empty (zero-volume) tensors, in the situation where they actually show up.

A detector that filters boxes by score produces **no boxes at all** on plenty of real frames, and
a serving stack that batches requests can hand a model a batch of zero rows. Both give TensorRT a
tensor with a 0 in its shape, and TensorRT handles them: the interesting part is the three ways a
program around it can get them wrong.

The cases below are a score-threshold post-processing chain (`Greater` -> `NonZero` -> `Gather`),
run once where nothing passes the threshold and once where something does, so the empty result can
be compared against the normal one rather than admired on its own.

Related examples: `02-API/Layer/NonZero` (the layer itself), `04-Feature/OutputAllocator`
(data-dependent shapes in general), `08-Advance/MultiOptimizationProfile` (profile ranges).
"""

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import TRTWrapperDDS, TRTWrapperV1, case_mark

N_BOX = 6
# One frame of "detections": six candidate boxes and their scores.
box_data = np.arange(N_BOX * 4, dtype=np.float32).reshape(N_BOX, 4)
score_data = np.array([0.10, 0.20, 0.05, 0.30, 0.15, 0.02], dtype=np.float32)

def build_filter_network(tw, threshold: float):
    """`score > threshold` -> `NonZero` -> `Gather`, i.e. the tail of any detector.

    The number of surviving boxes is data dependent, so the output shape is only known at run time
    and the engine has to be run through an output allocator (`TRTWrapperDDS`).
    """
    score = tw.network.add_input("score", trt.float32, [N_BOX])
    box = tw.network.add_input("box", trt.float32, [N_BOX, 4])

    threshold_tensor = tw.network.add_constant([1], np.array([threshold], dtype=np.float32))
    keep = tw.network.add_elementwise(score, threshold_tensor.get_output(0), trt.ElementWiseOperation.GREATER)
    index = tw.network.add_non_zero(keep.get_output(0), trt.DataType.INT32)  # [1, n_keep]
    flat_index = tw.network.add_shuffle(index.get_output(0))  # [n_keep]
    flat_index.reshape_dims = [-1]
    gather = tw.network.add_gather(box, flat_index.get_output(0), 0)  # [n_keep, 4]
    gather.get_output(0).name = "kept_box"
    return [gather.get_output(0)]

@case_mark
def case_no_detection():
    """The frame where nothing passes the threshold: the engine still runs and returns `[0, 4]`.

    This is the reassuring half of the story. No host-side `if n_detection == 0: skip` is needed:
    the same engine, the same call, only the reported shape differs.
    """
    for threshold in [0.5, 0.14]:
        tw = TRTWrapperDDS()
        tw.build(build_filter_network(tw, threshold))
        tw.setup({"score": score_data, "box": box_data}, b_print_io=False)
        tw.infer(b_print_io=False)
        kept_box = tw.buffer["kept_box"][0]
        print(f"    threshold={threshold}: kept_box shape={tuple(kept_box.shape)}, values={kept_box.reshape(-1)}")
    print("    -> an empty result is a normal result, not an error")
    return

@case_mark
def case_empty_tensor_needs_a_valid_address():
    """The trap: a zero-byte tensor still needs a non-null device address.

    `cudaMalloc(0)` succeeds and hands back **address 0**. Binding that as an input tensor makes
    `enqueueV3` refuse to run, and since it reports the refusal only through its return value, a
    program that ignores it reads the untouched output buffer and believes it got an answer.
    """
    logger = trt.Logger(trt.Logger.Severity.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network()
    # `previous` is the empty one (e.g. the detections carried over from the last frame),
    # `current` holds two boxes, and the two are concatenated.
    previous = network.add_input("previous", trt.float32, [-1, 2])
    current = network.add_input("current", trt.float32, [-1, 2])
    concatenation = network.add_concatenation([previous, current])
    concatenation.axis = 0
    concatenation.get_output(0).name = "merged"
    network.mark_output(concatenation.get_output(0))
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    for name in ["previous", "current"]:
        profile.set_shape(name, [0, 2], [4, 2], [8, 2])  # The minimum has to be 0, see the next case
    config.add_optimization_profile(profile)
    engine = trt.Runtime(logger).deserialize_cuda_engine(builder.build_serialized_network(network, config))

    status, null_address = cudart.cudaMalloc(0)
    print(f"    cudaMalloc(0) -> status={status}, address={null_address}  (success, and the address is NULL)")

    current_data = np.array([[0, 1], [2, 3]], dtype=np.float32)
    for b_use_null in [True, False]:
        context = engine.create_execution_context()
        context.set_input_shape("previous", [0, 2])
        context.set_input_shape("current", current_data.shape)

        # The sentinel makes an output that was never written distinguishable from a computed one.
        merged = np.full([2, 2], -999, dtype=np.float32)
        address_current = cudart.cudaMalloc(current_data.nbytes)[1]
        address_merged = cudart.cudaMalloc(merged.nbytes)[1]
        cudart.cudaMemcpy(address_merged, merged.ctypes.data, merged.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        cudart.cudaMemcpy(address_current, current_data.ctypes.data, current_data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        # The only difference between the two runs: what the empty input is bound to.
        address_previous = 0 if b_use_null else cudart.cudaMalloc(1)[1]

        context.set_tensor_address("previous", address_previous)
        context.set_tensor_address("current", address_current)
        context.set_tensor_address("merged", address_merged)
        b_enqueue = context.execute_async_v3(0)
        cudart.cudaDeviceSynchronize()
        cudart.cudaMemcpy(merged.ctypes.data, address_merged, merged.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        print(f"    empty input bound to {'NULL (cudaMalloc(0))' if b_use_null else 'a 1-byte allocation'}: "
              f"enqueueV3={b_enqueue}, merged={merged.reshape(-1)}")
        for address in [address_current, address_merged] + ([] if b_use_null else [address_previous]):
            cudart.cudaFree(address)

    print("    -> allocate at least 1 byte for a zero-volume tensor, and check what enqueueV3 returns:")
    print("       the failing run above still produced a plausible-looking output buffer")
    return

@case_mark
def case_reduce_over_empty_axis():
    """The quiet one: reducing over 0 elements returns the identity of the operation.

    SUM gives 0, but MAX gives `-inf` and AVG gives `NaN`, and those two travel: "the average score
    of this frame" is `NaN` for every frame with no detection, and one `NaN` poisons every later
    average, comparison and loss it touches. The fix is upstream (branch on the count) rather than
    downstream, because by then the value is indistinguishable from a genuine `NaN`.

    Note how this measurement depends on the previous case: taken with a NULL-bound empty input,
    the run never executes, the output buffer is never written, and the three numbers read back as
    plain zeros -- which is exactly the wrong conclusion, and is what this example first printed.
    """
    for n_row, tag in [(0, "empty"), (4, "4 rows")]:
        tw = TRTWrapperV1()
        tensor = tw.network.add_input("score", trt.float32, [-1, 3])
        tw.profile.set_shape(tensor.name, [0, 3], [4, 3], [8, 3])
        output_list = []
        for operation in [trt.ReduceOperation.SUM, trt.ReduceOperation.MAX, trt.ReduceOperation.AVG]:
            layer = tw.network.add_reduce(tensor, operation, 1 << 0, False)  # Reduce along the row axis
            layer.get_output(0).name = str(operation).split(".")[-1].lower()
            output_list.append(layer.get_output(0))
        tw.build(output_list)
        tw.setup({"score": np.arange(n_row * 3, dtype=np.float32).reshape(n_row, 3)}, b_print_io=False)
        tw.infer(b_print_io=False)
        result = {name: buffer[0] for name, buffer in tw.buffer.items() if name != "score"}
        print(f"    {tag:6s}: " + ", ".join(f"{name}={value}" for name, value in result.items()))
    print("    -> SUM over nothing is 0, but MAX is -inf and AVG is NaN: check the count before the reduction")
    return

@case_mark
def case_profile_must_cover_zero():
    """A shape of 0 is only legal if the optimization profile says so, and the check is silent.

    `set_input_shape` reports the rejection through its return value as well, so the same "nobody
    reads return codes" failure applies: the run continues with the previous shape.
    """
    for min_row in [1, 0]:
        logger = trt.Logger(trt.Logger.Severity.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network()
        tensor = network.add_input("x", trt.float32, [-1, 2])
        layer = network.add_identity(tensor)
        layer.get_output(0).name = "y"
        network.mark_output(layer.get_output(0))
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        profile.set_shape("x", [min_row, 2], [4, 2], [8, 2])
        config.add_optimization_profile(profile)
        engine = trt.Runtime(logger).deserialize_cuda_engine(builder.build_serialized_network(network, config))

        context = engine.create_execution_context()
        b_accepted = context.set_input_shape("x", [0, 2])
        print(f"    profile minimum [{min_row}, 2]: set_input_shape([0, 2]) -> {b_accepted}, "
              f"context shape of y = {tuple(context.get_tensor_shape('y'))}")
    print("    -> a model that may be fed an empty batch needs `min = 0` in its profile")
    return

if __name__ == "__main__":
    case_no_detection()
    case_empty_tensor_needs_a_valid_address()
    case_reduce_over_empty_axis()
    case_profile_must_cover_zero()

    print("\nFinish")
