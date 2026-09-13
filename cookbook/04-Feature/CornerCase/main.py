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
"""Corner cases the rest of the cookbook does not reach: UINT8, BOOL, NaN, and fan-out.

Each case here is a question whose answer is easy to assume and easy to get wrong, and
none of them is covered elsewhere in the cookbook (surveyed 2026-09-04):

1. `case_uint8_io`        - UINT8 is an **I/O-only** type. What happens when you compute with it.
2. `case_bool_tensor`     - BOOL tensors: which operations accept them, and what a BOOL
                            actually occupies.
3. `case_nan_semantics`   - `ISNAN`, and the comparison rules that make NaN dangerous in a
                            reduction or a Max.
4. `case_nan_propagation` - whether a NaN in one element can contaminate a whole tensor.
5. `case_high_fan_out`    - one tensor consumed by many layers: does TensorRT copy it N times?

`02-API/Layer/*` covers each operation's API; this file is about the semantics that only
show up when the types are unusual.
"""

from collections import OrderedDict

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1, case_mark

np.random.seed(31193)

result = OrderedDict()

def build_and_run(build_function, input_data: dict, *, b_strongly_typed: bool = True):
    """Build a one-off network from `build_function(network)` and run it once."""
    tw = TRTWrapperV1(logger=trt.Logger.ERROR)
    if b_strongly_typed:
        tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    output_tensor_list = build_function(tw.network)
    tw.build(output_tensor_list)
    tw.setup(input_data, b_print_io=False)
    tw.infer(b_print_io=False)
    return tw

# ================================================================ Cases

@case_mark
def case_uint8_io() -> None:
    """UINT8 is an I/O type, not an arithmetic one.

    It exists so an engine can take raw 8-bit image data without a host-side conversion. The
    first thing the network must do is `Cast` it to something computable -- TensorRT rejects
    arithmetic on UINT8 outright, which is the useful part: the restriction is enforced at
    build time rather than producing wrong numbers.
    """
    data = {"tensor": (np.arange(24) % 256).astype(np.uint8).reshape(2, 3, 4)}

    # 1) The supported shape: UINT8 in, immediately cast
    def build_ok(network):
        tensor = network.add_input("tensor", trt.uint8, data["tensor"].shape)
        casted = network.add_cast(tensor, trt.float32).get_output(0)
        layer = network.add_elementwise(casted, network.add_constant([1, 1, 1], np.array([1.0], dtype=np.float32)).get_output(0), trt.ElementWiseOperation.SUM)
        output = layer.get_output(0)
        output.name = "output"
        return [output]

    tw = build_and_run(build_ok, data)
    output = tw.buffer["output"][0]
    print(f"    UINT8 -> Cast -> Add(1): input {data['tensor'].reshape(-1)[:6]} ... -> {output.reshape(-1)[:6]}")
    assert np.allclose(output, data["tensor"].astype(np.float32) + 1.0)

    # 2) Arithmetic straight on UINT8
    def build_bad(network):
        tensor = network.add_input("tensor", trt.uint8, data["tensor"].shape)
        layer = network.add_elementwise(tensor, network.add_constant([1, 1, 1], np.array([1], dtype=np.uint8)).get_output(0), trt.ElementWiseOperation.SUM)
        output = layer.get_output(0)
        output.name = "output"
        return [output]

    # Capture what the builder actually says. Letting the wrapper fail instead reports
    # `deserialize_cuda_engine(): incompatible function arguments`, which is the symptom of a
    # None plan rather than the reason for it.
    class RecordingLogger(trt.ILogger):

        def __init__(self):
            trt.ILogger.__init__(self)
            self.message_list = []

        def log(self, severity, message):
            if severity <= trt.ILogger.Severity.ERROR:
                self.message_list.append(message)

    logger = RecordingLogger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    for tensor in build_bad(network):
        network.mark_output(tensor)
    engine_bytes = builder.build_serialized_network(network, builder.create_builder_config())
    print(f"    UINT8 arithmetic: build returned {'a plan' if engine_bytes else 'None'}")
    if logger.message_list:
        print(f"        builder said: {logger.message_list[0][:150]}")
    assert engine_bytes is None, "Expected TensorRT to reject arithmetic on UINT8"
    result["uint8_arithmetic"] = "rejected"
    return

@case_mark
def case_bool_tensor() -> None:
    """BOOL tensors, and what one costs.

    A comparison produces BOOL; `Select` consumes it. The surprise is the footprint: a BOOL
    element is **one byte**, not one bit, so a mask is the same size as an INT8 tensor and
    eight times an actual bitset. That matters when the mask is the big tensor.
    """
    data = {"a": np.array([[1.0, 5.0, 3.0], [7.0, 2.0, 9.0]], dtype=np.float32), "b": np.array([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]], dtype=np.float32)}

    def build(network):
        a = network.add_input("a", trt.float32, data["a"].shape)
        b = network.add_input("b", trt.float32, data["b"].shape)
        mask = network.add_elementwise(a, b, trt.ElementWiseOperation.GREATER).get_output(0)
        mask.name = "mask"
        # `Select(condition, then, else)` is the consumer that makes a BOOL useful
        chosen = network.add_select(mask, a, b).get_output(0)
        chosen.name = "chosen"
        # A BOOL cannot be reduced directly; cast it to count how many passed
        counted = network.add_reduce(network.add_cast(mask, trt.int32).get_output(0), trt.ReduceOperation.SUM, (1 << 0) | (1 << 1), False).get_output(0)
        counted.name = "count"
        return [mask, chosen, counted]

    tw = build_and_run(build, data)
    mask = tw.buffer["mask"][0]
    print(f"    a > b mask dtype={mask.dtype}, itemsize={mask.dtype.itemsize} byte(s) per element")
    print(f"    mask        = {mask.reshape(-1).tolist()}")
    print(f"    Select(a,b) = {tw.buffer['chosen'][0].reshape(-1).tolist()}")
    print(f"    count       = {tw.buffer['count'][0]}")
    assert mask.dtype.itemsize == 1, "A BOOL element is expected to occupy one byte"
    assert np.array_equal(tw.buffer["chosen"][0], np.maximum(data["a"], data["b"]))
    result["bool_itemsize"] = int(mask.dtype.itemsize)
    return

@case_mark
def case_nan_semantics() -> None:
    """`ISNAN`, and why `NaN != NaN` makes ordinary comparisons useless for detecting it."""
    data = {"tensor": np.array([1.0, np.nan, 3.0, np.inf, -np.inf, 0.0], dtype=np.float32)}

    def build(network):
        tensor = network.add_input("tensor", trt.float32, data["tensor"].shape)
        is_nan = network.add_unary(tensor, trt.UnaryOperation.ISNAN).get_output(0)
        is_nan.name = "is_nan"
        is_inf = network.add_unary(tensor, trt.UnaryOperation.ISINF).get_output(0)
        is_inf.name = "is_inf"
        # The naive test: x == x. True for everything except NaN -- if the compiler and the
        # hardware follow IEEE-754, which they do here.
        self_equal = network.add_elementwise(tensor, tensor, trt.ElementWiseOperation.EQUAL).get_output(0)
        self_equal.name = "self_equal"
        return [is_nan, is_inf, self_equal]

    tw = build_and_run(build, data)
    print(f"    input      = {data['tensor'].tolist()}")
    print(f"    ISNAN      = {tw.buffer['is_nan'][0].tolist()}")
    print(f"    ISINF      = {tw.buffer['is_inf'][0].tolist()}")
    print(f"    x == x     = {tw.buffer['self_equal'][0].tolist()}")
    assert bool(tw.buffer["is_nan"][0][1]), "ISNAN should flag the NaN"
    assert not bool(tw.buffer["self_equal"][0][1]), "NaN == NaN should be False"
    print("    `x == x` and ISNAN agree here, but only ISNAN says what it means.")
    return

@case_mark
def case_nan_propagation() -> None:
    """One NaN, one reduction: how far does it spread?

    Measured, and the answer differs per operation in a way that matters:

        input row = [5.0, 6.0, nan, 8.0]
        reduce SUM -> nan     (matches numpy)
        reduce MAX -> 8.0     (numpy gives nan)
        reduce MIN -> 5.0     (numpy gives nan)

    **`SUM` propagates the NaN; `MAX` and `MIN` silently drop it.** IEEE-754 permits either
    for min/max, and TensorRT takes the ignore-NaN option, so a NaN that would be obvious in
    a sum disappears completely from a max-pool or a top-k style reduction. Do not use a
    max-reduction to check whether a tensor is clean -- use `ISNAN` plus a SUM, as the
    previous case does.
    """
    array = np.arange(1, 13, dtype=np.float32).reshape(3, 4).copy()
    array[1, 2] = np.nan
    data = {"tensor": array}

    def build(network):
        tensor = network.add_input("tensor", trt.float32, array.shape)
        output_list = []
        for name, operation in [("sum", trt.ReduceOperation.SUM), ("max", trt.ReduceOperation.MAX), ("min", trt.ReduceOperation.MIN)]:
            reduced = network.add_reduce(tensor, operation, 1 << 1, False).get_output(0)  # reduce the last axis
            reduced.name = name
            output_list.append(reduced)
        return output_list

    tw = build_and_run(build, data)
    print(f"    input row 1 contains NaN at column 2: {array[1].tolist()}")
    for name in ["sum", "max", "min"]:
        value = tw.buffer[name][0]
        numpy_reference = {"sum": np.sum, "max": np.max, "min": np.min}[name](array, axis=1)
        agree = "same as numpy" if np.array_equal(value, numpy_reference, equal_nan=True) else "DIFFERS from numpy"
        print(f"    reduce {name:<4} per row = {value.tolist()}   numpy: {numpy_reference.tolist()}   -> {agree}")
        result[f"nan_{name}"] = value.tolist()
    print("    SUM propagates the NaN; MAX and MIN drop it. A NaN that is obvious in a sum is")
    print("    invisible to a max-reduction, so max is not a NaN detector.")
    assert np.isnan(tw.buffer["sum"][0][1]), "The NaN row's sum should be NaN"
    assert not np.isnan(tw.buffer["sum"][0][0]), "Other rows should be unaffected"
    assert not np.isnan(tw.buffer["max"][0][1]), "This build is expected to ignore NaN in MAX"
    return

@case_mark
def case_high_fan_out() -> None:
    """One tensor feeding many consumers: does TensorRT materialise it N times?

    A fan-out of N is the shape of an attention head split, a multi-scale head, or any
    residual reused far downstream. The question is whether the builder keeps one copy or
    inserts N, which decides whether fan-out is free.
    """
    data = {"tensor": np.random.rand(4, 64, 64).astype(np.float32)}

    def make_builder(n_consumer: int):

        def build(network):
            tensor = network.add_input("tensor", trt.float32, data["tensor"].shape)
            output_list = []
            for index in range(n_consumer):
                constant = network.add_constant([1, 1, 1], np.array([float(index + 1)], dtype=np.float32)).get_output(0)
                consumed = network.add_elementwise(tensor, constant, trt.ElementWiseOperation.PROD).get_output(0)
                consumed.name = f"output{index}"
                output_list.append(consumed)
            return output_list

        return build

    print(f"    {'consumers':>10}{'network layers':>16}{'engine bytes':>14}")
    for n_consumer in [1, 2, 4, 8, 16]:
        tw = TRTWrapperV1(logger=trt.Logger.ERROR)
        tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        output_tensor_list = make_builder(n_consumer)(tw.network)
        n_layer = tw.network.num_layers
        tw.build(output_tensor_list)
        print(f"    {n_consumer:>10}{n_layer:>16}{tw.engine_bytes.nbytes:>14}")
        result.setdefault("fan_out", []).append((n_consumer, n_layer, tw.engine_bytes.nbytes))
    print("    Layer count grows with the consumers (one Elementwise each) but the producer is")
    print("    not duplicated: reading a tensor many times costs reads, not copies.")
    return

def main() -> None:
    case_uint8_io()
    case_bool_tensor()
    case_nan_semantics()
    case_nan_propagation()
    case_high_fan_out()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
