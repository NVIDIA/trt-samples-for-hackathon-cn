#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
#

from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import (NetworkSerialization, TRTWrapperV2, case_mark, check_array, datatype_np_to_trt)

json_file = Path("./unit-tests-network.json")
para_file = Path("./unit-tests-network.npz")

# Tool function
def test_single_layer(tw, output_tensor_list, data, expect_fail_building=False):
    is_pass = True

    print("Run baseline")
    tw.build(output_tensor_list)
    if not expect_fail_building:
        tw.setup(data, b_print_io=True)
        tw.infer(b_print_io=True)
        output_ref = {name: tw.buffer[name][0] for name in tw.buffer.keys() if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT}

    print("Serialization")
    ns = NetworkSerialization(json_file, para_file)
    ns.serialize(logger=tw.logger, builder=tw.builder, builder_config=tw.config, network=tw.network, optimization_profile_list=[tw.profile])

    print("Deserialization")
    ns2 = NetworkSerialization(json_file, para_file)
    ns2.deserialize(logger=tw.logger)

    tw2 = TRTWrapperV2(logger=tw.logger)
    tw2.builder = ns2.builder
    tw2.network = ns2.network
    tw2.config = ns2.builder_config

    print("Build and run")
    tw2.build()
    if not expect_fail_building:
        tw2.setup(data, b_print_io=True)
        tw2.infer(b_print_io=True)
        output_rebuild = {name: tw.buffer[name][0] for name in tw.buffer.keys() if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT}

        for name in output_ref.keys():
            check_array(output_rebuild[name], output_ref[name], des=name)

    return is_pass

@case_mark
def test_activation_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.arange(9, dtype=np.float32).reshape(3, 3) - 4}  # [0, 8] -> [-4, 4]}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            # case_simple
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_activation(tensor, trt.ActivationType.SCALED_TANH)
            layer.alpha = -2
            layer.beta = 2

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_assert_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.ones([3, 4, 5], dtype=np.float32)}
    data1 = {"tensor": np.ones([3, 4, 5], dtype=np.float32), "tensor1": np.ones([3, 4], dtype=np.float32)}
    data2 = {"tensor": np.ones([3, 4, 5], dtype=np.float32), "tensor1": np.ones([3, 5], dtype=np.float32)}

    return is_pass  # Disable it since we expect it throw exception

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            # case buildtime check
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer1 = tw.network.add_shape(tensor)
            layer2 = tw.network.add_slice(layer1.get_output(0), [2], [1], [1])
            layerConstant = tw.network.add_constant([1], np.array([4], dtype=np.int64))
            layer3 = tw.network.add_elementwise(layer2.get_output(0), layerConstant.get_output(0), trt.ElementWiseOperation.EQUAL)
            layer4 = tw.network.add_identity(layer3.get_output(0))
            layer4.get_output(0).dtype = trt.bool
            layer = tw.network.add_assertion(layer4.get_output(0), "tensor.shape[2] != 5")
            layer.message += " [Something else you want to say]"  # [Optional] Reset assert message later
            layer_output = layer4
            input_data = data
        elif i == 1:
            # case runtime check
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), [-1, -1, -1])
            tw.profile.set_shape(tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), [-1, -1])
            tw.profile.set_shape(tensor1.name, [1, 1], [3, 4], [6, 8])
            tw.config.add_optimization_profile(tw.profile)
            layer1 = tw.network.add_shape(tensor)
            layer2 = tw.network.add_slice(layer1.get_output(0), [1], [1], [1])
            layer3 = tw.network.add_shape(tensor1)
            layer4 = tw.network.add_slice(layer3.get_output(0), [1], [1], [1])
            layer5 = tw.network.add_elementwise(layer2.get_output(0), layer4.get_output(0), trt.ElementWiseOperation.EQUAL)
            layer = tw.network.add_assertion(layer5.get_output(0), "tensor.shape[1] != tensor1.shape[1]")
            layer6 = tw.network.add_cast(layer5.get_output(0), trt.int32)
            layer_output = layer6
            input_data = data2
        try:
            test_single_layer(tw, [layer_output.get_output(0)], input_data)
            is_pass = is_pass and False
        except:
            is_pass = is_pass and True
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_cast_layer():
    n_test_case = 2
    is_pass = True
    # Common data
    data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(3, 4, 5) * 10 - 300}  # [0,59] -> [-300, 290]

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tw.config.set_flag(trt.BuilderFlag.FP16)  # Need this if using float16, similarly BF16 for bfloat16
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_cast(tensor, trt.DataType.HALF)
            layer.get_output(0).dtype = trt.DataType.HALF
            layer1 = tw.network.add_cast(tensor, trt.DataType.INT32)
            layer2 = tw.network.add_cast(tensor, trt.uint8)
            output_tensor_list = [layer.get_output(0), layer1.get_output(0), layer2.get_output(0)]
        elif i == 1:
            tw.config.set_flag(trt.BuilderFlag.INT8)
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_cast(tensor, trt.int8)
            layer.get_input(0).dynamic_range = [-300, 300]
            layer.get_output(0).dynamic_range = [-300, 300]
            layer.get_output(0).dtype = trt.DataType.INT8
            output_tensor_list = [layer.get_output(0)]

        is_pass = is_pass and test_single_layer(tw, output_tensor_list, data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_concatenation_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(3, 4, 5)}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            # case_simple
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_concatenation([tensor, tensor])
            layer.axis = 2

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)

    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_constant_layer():
    n_test_case = 2
    is_pass = True
    # Common data
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}
    data1 = {
        "tensor": np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [-1, -2, -3, -4, -5, -6, -7, -8],
            [7, 6, 5, 4, 3, 2, 1, 0],
            [-7, -6, -5, -4, -3, -2, -1, 0],
        ], dtype=np.int8)
    }

    def pack_int4(array: np.ndarray):  # copy from https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/Constant.html
        result = []
        array = array.flatten()
        for low, high in zip(array[::2], array[1::2]):
            low = np.rint(np.clip(low, -8, 7)).astype(np.int8)
            high = np.rint(np.clip(high, -8, 7)).astype(np.int8)
            result.append(high << 4 | low & 0x0F)
        return np.asarray(result, dtype=np.int8)

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            layer = tw.network.add_constant(data["tensor"].shape, trt.Weights(np.ascontiguousarray(data["tensor"])))

        elif i == 1:
            continue  # Disable this since  we can not get value from `trt.Weights`
            data1_packed = pack_int4(data1["tensor"])
            layer = tw.network.add_constant(data1["tensor"].shape, weights=trt.Weights(trt.int4, data1_packed.ctypes.data, data1["tensor"].size))
            layer1 = tw.network.add_constant(shape=(), weights=np.ones(shape=(1), dtype=np.float32))
            layer = tw.network.add_dequantize(layer.get_output(0), layer1.get_output(0), trt.float32)
            layer.precision = trt.int4

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], {})
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_convolution_layer():
    n_test_case = 5
    is_pass = True
    # Common data
    n_b, n_c, n_h, n_w = [1, 1, 6, 9]
    n_cout, n_hk, n_wk = [1, 3, 3]  # Number of output channel, kernel height and kernel width
    data = np.arange(n_hk * n_wk, dtype=np.float32).reshape(1, 1, n_hk, n_wk)
    data = np.tile(data, (n_b, n_c, n_h // n_hk, n_w // n_wk)) + 1
    data = {"tensor": data}
    w = np.ascontiguousarray(np.power(10, range(4, -5, -1), dtype=np.float32).reshape(n_cout, n_c, n_hk, n_wk))
    b = np.ascontiguousarray(np.zeros(n_cout, dtype=np.float32))

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            # case_simple
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_convolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))
            input_data = data
        elif i == 1:
            # case_stride_dilation_pad
            nHStride, nWStride = 2, 2
            nHDilation, nWDilation = 2, 2
            nHPadding, nWPadding = 1, 1
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_convolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))
            layer.stride_nd = [nHStride, nWStride]
            layer.dilation_nd = [nHDilation, nWDilation]
            layer.padding_nd = [nHPadding, nWPadding]
            layer.pre_padding = [nHPadding, nWPadding]
            layer.post_padding = [nHPadding, nWPadding]
            layer.padding_mode = trt.PaddingMode.SAME_UPPER
            input_data = data
        elif i == 2:
            # case_group
            n_cout1 = 2
            n_group = 2
            data1 = {"tensor": np.tile(data["tensor"], [1, n_cout1 // n_c, 1, 1])}
            w1 = np.ascontiguousarray(np.concatenate([w, -w], 0))
            b1 = np.ascontiguousarray(np.zeros(n_cout1, dtype=np.float32))
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
            layer = tw.network.add_convolution_nd(tensor, n_cout1, [n_hk, n_wk], trt.Weights(w1), trt.Weights(b1))
            layer.num_groups = n_group
            input_data = data1
        elif i == 3:
            # case 3D
            n_c1 = 2
            data1 = {"tensor": np.tile(data["tensor"], [1, n_c1 // n_c, 1, 1]).reshape([n_b, 1, n_c1, n_h, n_w])}
            w1 = np.ascontiguousarray(np.concatenate([w, -w], 0))
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
            layer = tw.network.add_convolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w1), trt.Weights(b))
            input_data = data1
        elif i == 4:
            # case_int8qdq
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer_q0_weight = tw.network.add_constant([], np.array([1], dtype=np.float32))
            layer_q1_weight = tw.network.add_constant([], np.array([1], dtype=np.float32))
            layer_weight = tw.network.add_constant(w.shape, trt.Weights(w))
            layer_q0 = tw.network.add_quantize(tensor, layer_q0_weight.get_output(0))
            layer_q0.axis = 0
            layer_dq0 = tw.network.add_dequantize(layer_q0.get_output(0), layer_q1_weight.get_output(0))
            layer_dq0.axis = 0
            layer_q1 = tw.network.add_quantize(layer_weight.get_output(0), layer_q0_weight.get_output(0))
            layer_q1.axis = 0
            layer_dq1 = tw.network.add_dequantize(layer_q1.get_output(0), layer_q1_weight.get_output(0))
            layer_dq1.axis = 0
            layer = tw.network.add_convolution_nd(layer_dq0.get_output(0), n_cout, [n_hk, n_wk], trt.Weights(), trt.Weights(b))
            layer.set_input(1, layer_dq1.get_output(0))  # Set weight from tensor rather than constructor
            input_data = data

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], input_data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_cumulative_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer_axis = tw.network.add_constant(shape=(), weights=np.array([1], dtype=np.int32))
            layer = tw.network.add_cumulative(tensor, layer_axis.get_output(0), trt.CumulativeOperation.SUM, False, False)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)

    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_deconvolution_layer():
    n_test_case = 5
    is_pass = True
    # Common data
    n_b, n_c, n_h, n_w = [1, 1, 3, 3]
    n_cout, n_hk, n_wk = [1, 3, 3]  # Number of output channel, kernel height and kernel width
    data = np.arange(np.prod(n_b * n_c * n_h * n_w), dtype=np.float32).reshape(n_b, n_c, n_h, n_w) + 1
    data = {"tensor": data}
    w = np.ascontiguousarray(np.power(10, range(4, -5, -1), dtype=np.float32).reshape(n_cout, n_c, n_hk, n_wk))
    b = np.ascontiguousarray(np.zeros(n_cout, dtype=np.float32))

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            # case_simple
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_deconvolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))
            input_data = data
        elif i == 1:
            # case_stride_dilation_pad
            nHStride, nWStride = 2, 2
            nHDilation, nWDilation = 2, 2
            nHPadding, nWPadding = 1, 1
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_deconvolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))
            layer.stride_nd = [nHStride, nWStride]
            layer.dilation_nd = [nHDilation, nWDilation]
            layer.padding_nd = [nHPadding, nWPadding]
            layer.pre_padding = [nHPadding, nWPadding]
            layer.post_padding = [nHPadding, nWPadding]
            layer.padding_mode = trt.PaddingMode.SAME_UPPER
            input_data = data
        elif i == 2:
            # case_group
            n_cout1 = 2  # n_c in this example is 2
            n_group = 2
            data1 = {"tensor": np.tile(data["tensor"], [1, n_cout1 // n_c, 1, 1])}
            w1 = np.ascontiguousarray(np.concatenate([w, -w], 0))  # double the kernel as shape of [n_group, n_hk, n_wk]
            b1 = np.ascontiguousarray(np.zeros(n_cout1, dtype=np.float32))
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
            layer = tw.network.add_deconvolution_nd(tensor, n_cout1, [n_hk, n_wk], trt.Weights(w1), trt.Weights(b1))
            layer.num_groups = n_group
            input_data = data1
        elif i == 3:
            # case 3D
            n_c1 = 2
            data1 = {"tensor": np.tile(data["tensor"], [1, n_c1 // n_c, 1, 1]).reshape([n_b, 1, n_c1, n_h, n_w])}
            w1 = np.ascontiguousarray(np.concatenate([w, -w], 0))
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
            layer = tw.network.add_deconvolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w1), trt.Weights(b))
            input_data = data1
        elif i == 4:
            # case_int8qdq
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer_q0_weight = tw.network.add_constant([], np.array([1], dtype=np.float32))
            layer_q1_weight = tw.network.add_constant([], np.array([1], dtype=np.float32))
            layer_weight = tw.network.add_constant(w.shape, trt.Weights(w))
            layer_q0 = tw.network.add_quantize(tensor, layer_q0_weight.get_output(0))
            layer_q0.axis = 0
            layer_dq0 = tw.network.add_dequantize(layer_q0.get_output(0), layer_q1_weight.get_output(0))
            layer_dq0.axis = 0
            layer_q1 = tw.network.add_quantize(layer_weight.get_output(0), layer_q0_weight.get_output(0))
            layer_q1.axis = 0
            layer_dq1 = tw.network.add_dequantize(layer_q1.get_output(0), layer_q1_weight.get_output(0))
            layer_dq1.axis = 0
            layer = tw.network.add_deconvolution_nd(layer_dq0.get_output(0), n_cout, [n_hk, n_wk], trt.Weights(), trt.Weights(np.ascontiguousarray(b)))  # weight as empty
            layer.set_input(1, layer_dq1.get_output(0))
            input_data = data

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], input_data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_dynamic_quantize_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": (np.arange(48, dtype=np.float32)).reshape(3, 16) / 24 - 1}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            double_quantization_layer = tw.network.add_constant(shape=[], weights=np.array([1], dtype=np.float32))
            layer = tw.network.add_dynamic_quantize(tensor, 1, 16, trt.DataType.FP4, trt.DataType.FP8)
            layer.set_input(1, double_quantization_layer.get_output(0))

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_einsum_layer():
    n_test_case = 5
    is_pass = True

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            data0 = np.arange(np.prod(12), dtype=np.float32).reshape(1, 3, 4)
            data1 = np.arange(np.prod(30), dtype=np.float32).reshape(2, 3, 5)
            data = {"tensor": data0, "tensor1": data1}
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_einsum([tensor, tensor1], "ijk,pjr->ikpr")
        elif i == 1:
            data = {"tensor": np.arange(np.prod(12), dtype=np.float32).reshape(1, 3, 4)}
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_einsum([tensor], "ijk->jki")
        elif i == 2:
            data = {"tensor": np.arange(np.prod(12), dtype=np.float32).reshape(1, 3, 4)}
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_einsum([tensor], "ijk->ij")
        elif i == 3:
            shape0 = 1, 2, 4
            shape1 = 1, 3, 4
            equation = "ijk,pqk->j"
            data0 = np.arange(np.prod(shape0), dtype=np.float32).reshape(shape0)
            data1 = np.ones(np.prod(shape1), dtype=np.float32).reshape(shape1)
            data = {"tensor": data0, "tensor1": data1}
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_einsum([tensor, tensor1], equation)
        elif i == 4:
            data0 = np.arange(np.prod(12), dtype=np.float32).reshape(2, 2, 3)
            data1 = np.ones(np.prod(24), dtype=np.float32).reshape(2, 3, 4)
            data = {"tensor": data0, "tensor1": data1}
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_einsum([tensor, tensor1], "ijk,ikl->ijl")

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_elementwise_layer():
    n_test_case = 2
    is_pass = True
    # Common data
    data0 = np.full([3, 4, 5], 2, dtype=np.float32)
    data1 = np.full([3, 4, 5], 3, dtype=np.float32)
    data = {"tensor": data0, "tensor1": data1}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_elementwise(tensor, tensor1, trt.ElementWiseOperation.SUM)
        elif i == 1:
            n_c, n_h, n_w = data["tensor"].shape
            data0 = np.full([n_c, 1, n_w], 1, dtype=np.float32)
            data1 = np.full([n_c, n_h, 1], 2, dtype=np.float32)
            data1 = {"tensor": data0, "tensor1": data1}
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), data1["tensor1"].shape)
            layer = tw.network.add_elementwise(tensor, tensor1, trt.ElementWiseOperation.SUM)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_qdq_structure():
    n_test_case = 3
    is_pass = True
    # Common data
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tw.config.set_flag(trt.BuilderFlag.INT8)
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer_q_scale = tw.network.add_constant([], np.array([60 / 127], dtype=np.float32))
            layer_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
            layer_q = tw.network.add_quantize(tensor, layer_q_scale.get_output(0))
            layer_dq = tw.network.add_dequantize(layer_q.get_output(0), layer_dq_scale.get_output(0))
        elif i == 1:
            tw.config.set_flag(trt.BuilderFlag.INT8)
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer_q_scale = tw.network.add_constant([4], np.array([40 / 127, 80 / 127, 120 / 127, 160 / 127], dtype=np.float32))
            layer_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
            layer_q = tw.network.add_quantize(tensor, layer_q_scale.get_output(0))
            layer_q.axis = 1
            layer_dq = tw.network.add_dequantize(layer_q.get_output(0), layer_dq_scale.get_output(0))
        elif i == 2:
            tw.config.set_flag(trt.BuilderFlag.INT8)
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer_q_scale = tw.network.add_constant([4], np.array([20 / 127, 40 / 127, 60 / 127, 80 / 127], dtype=np.float32))
            layer_q_zeropoint = tw.network.add_constant([4], np.array([0, 0, 0, 0], dtype=np.float32))
            layer_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
            layer_q = tw.network.add_quantize(tensor, layer_q_scale.get_output(0))
            layer_q.axis = 1
            layer_q.set_input(2, layer_q_zeropoint.get_output(0))
            layer_dq = tw.network.add_dequantize(layer_q.get_output(0), layer_dq_scale.get_output(0))
            layer_dq.axis = 0

        is_pass = is_pass and test_single_layer(tw, [layer_dq.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_fill_layer():
    n_test_case = 6
    is_pass = True

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            output_shape = [4]
            layer = tw.network.add_fill(output_shape, trt.FillOperation.LINSPACE, trt.DataType.FLOAT)
            layer.alpha = [1000]
            layer.beta = [1]
            data = {}
        elif i == 1:
            output_shape = [3, 4, 5]
            data0 = np.array(1000, dtype=np.float32)
            data1 = np.array([100, 10, 1], dtype=np.float32)
            data = {"tensor": data0, "tensor1": data1}
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_fill(output_shape, trt.FillOperation.LINSPACE, trt.DataType.FLOAT)
            layer.set_input(1, tensor)
            layer.set_input(2, tensor1)
        elif i == 2:  # Remove this case if varibale random numbers can be provided among buildings
            output_shape = [3, 4, 5]
            data0 = np.array(0, dtype=np.float32)
            data1 = np.array(0.92, dtype=np.float32)
            data = {"tensor": data0, "tensor1": data1}
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_fill(output_shape, trt.FillOperation.RANDOM_NORMAL)
            layer.set_input(1, tensor)
            layer.set_input(2, tensor1)
        elif i == 3:
            output_shape = [3, 4, 5]
            data0 = np.array(5, dtype=np.float32)
            data1 = np.array(10, dtype=np.float32)
            data = {"tensor": data0, "tensor1": data1}
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_fill(output_shape, trt.FillOperation.RANDOM_UNIFORM)
            layer.set_input(1, tensor)
            layer.set_input(2, tensor1)
        elif i == 4:
            output_shape = [3, 4, 5]
            data0 = np.array(output_shape, dtype=np.int32)
            data1 = np.float32(1000)
            data2 = np.array([100, 10, 1], dtype=np.float32)
            data = {"tensor": data0, "tensor1": data1, "tensor2": data2}
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
            tw.profile.set_shape_input(tensor.name, [1, 1, 1], output_shape, output_shape)  # Range of value rather than shape
            tw.config.add_optimization_profile(tw.profile)
            layer = tw.network.add_fill([], trt.FillOperation.LINSPACE)
            layer.set_input(0, tensor)  # Use index 0 to set output shape
            layer.set_input(1, tensor1)
            layer.set_input(2, tensor2)
        elif i == 5:
            data0 = np.zeros([3, 4, 5]).astype(np.float32)
            data0[0, 0, 1] = 1
            data0[0, 2, 3] = 2
            data0[0, 3, 4] = 3
            data0[1, 1, 0] = 4
            data0[1, 1, 1] = 5
            data0[1, 1, 2] = 6
            data0[1, 1, 3] = 7
            data0[1, 1, 4] = 8
            data0[2, 0, 1] = 9
            data0[2, 1, 1] = 10
            data0[2, 2, 1] = 11
            data0[2, 3, 1] = 12
            data1 = np.float32(1000)
            data2 = np.array([10, 1], dtype=np.float32)
            data = {"tensor": data0, "tensor1": data1, "tensor2": data2}
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), [-1 for _ in data["tensor"].shape])
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
            tw.profile.set_shape(tensor.name, [1, 1, 1], [3, 4, 5], [3, 4, 5])
            tw.config.add_optimization_profile(tw.profile)
            layer1 = tw.network.add_non_zero(tensor)
            layer2 = tw.network.add_shape(layer1.get_output(0))
            layer = tw.network.add_fill([], trt.FillOperation.LINSPACE)
            layer.set_input(0, layer2.get_output(0))  # Use index 0 to set output shape
            layer.set_input(1, tensor1)
            layer.set_input(2, tensor2)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_gather_layer():
    n_test_case = 6
    is_pass = True
    # Common data
    shape = [2, 3, 4, 5]
    data0 = np.arange(shape[0]).reshape(shape[0], 1, 1, 1) * 1000 + \
        np.arange(shape[1]).reshape(1, shape[1], 1, 1) * 100 + \
        np.arange(shape[2]).reshape(1, 1, shape[2], 1) * 10 + \
        np.arange(shape[3]).reshape(1, 1, 1, shape[3])
    data = {"tensor": data0.astype(np.float32)}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            data["tensor1"] = np.array([[1, 0, 2], [0, 0, -1]], dtype=np.int32)
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.DEFAULT)
            layer.axis = 2
        elif i == 1:
            data["tensor1"] = np.array([[1, 0, 2], [0, 0, -1]], dtype=np.int32)
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.DEFAULT)
            layer.axis = 2
            layer.num_elementwise_dims = 1
        elif i == 2:
            data1 = np.zeros(data0.shape, dtype=np.int32)
            # use random permutation
            for i in range(data0.shape[0]):
                for j in range(data0.shape[1]):
                    for k in range(data0.shape[3]):
                        data1[i, j, :, k] = np.random.permutation(range(data0.shape[2]))
            data["tensor1"] = data1
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.ELEMENT)
            layer.axis = 2
        elif i == 3:
            data["tensor1"] = np.array([[1, 0, 2], [0, 2, -1]], dtype=np.int32)
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.ND)
        elif i == 4:
            data["tensor1"] = np.array([[1, 0, 2], [0, 2, -1]], dtype=np.int32)
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.ND)
            layer.num_elementwise_dims = 1
        elif i == 5:
            data = np.zeros([3, 4, 5]).astype(np.float32)
            data[0, 0, 1] = 1
            data[0, 2, 3] = 2
            data[0, 3, 4] = 3
            data[1, 1, 0] = 4
            data[1, 1, 1] = 5
            data[1, 1, 2] = 6
            data[1, 1, 3] = 7
            data[1, 1, 4] = 8
            data[2, 0, 1] = 9
            data[2, 1, 1] = 10
            data[2, 2, 1] = 11
            data[2, 3, 1] = 12
            data = {"tensor": data}  # Change the common `data`
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_non_zero(tensor)
            layer = tw.network.add_shuffle(layer.get_output(0))
            layer.first_transpose = [1, 0]
            layer = tw.network.add_gather_v2(tensor, layer.get_output(0), trt.GatherMode.ND)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_grid_sample_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    shape = [1, 3, 4, 5]
    shape1 = [6, 10]
    data0 = np.arange(shape[0]).reshape(shape[0], 1, 1, 1) * 1000 + \
        np.arange(shape[1]).reshape(1, shape[1], 1, 1) * 100 + \
        np.arange(shape[2]).reshape(1, 1, shape[2], 1) * 10 + \
        np.arange(shape[3]).reshape(1, 1, 1, shape[3])
    data0 = data0.astype(np.float32)
    dataX = np.random.randint(0, shape[2], [shape[0], shape1[0], shape1[1], 1], dtype=np.int32) / (shape[2] - 1) * 2 - 1
    dataY = np.random.randint(0, shape[3], [shape[0], shape1[0], shape1[1], 1], dtype=np.int32) / (shape[3] - 1) * 2 - 1
    data1 = np.concatenate([dataX, dataY], axis=3).astype(np.float32)

    data = {"tensor": data0, "tensor1": data1}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_grid_sample(tensor, tensor1)
            layer.align_corners = False
            layer.interpolation_mode = trt.InterpolationMode.LINEAR
            layer.sample_mode = trt.SampleMode.FILL

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_identity_layer():
    n_test_case = 2
    is_pass = True
    # Common data
    data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(1, 3, 4, 5)}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_identity(tensor)
            output_tensor_list = [layer.get_output(0)]
        elif i == 1:
            tw.config.set_flag(trt.BuilderFlag.FP16)  # Needed if using float16
            tw.config.set_flag(trt.BuilderFlag.BF16)  # Needed if using bfloat16
            tw.config.set_flag(trt.BuilderFlag.INT8)  # Needed if using int8

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            output_tensor_list = []
            for data_type in [trt.float16, trt.bfloat16, trt.int32, trt.int64, trt.int8, trt.uint8, trt.bool]:  # trt.int4
                # FP8 / FP4 is only supported from Plugin / Quantize / Constant / Concatenation / Shuffle layer
                layer = tw.network.add_identity(tensor)
                layer.set_output_type(0, data_type)
                if data_type == trt.int8:
                    layer.get_output(0).set_dynamic_range(0, 127)  # Dynamic range or calibration needed for INT8
                output_tensor_list.append(layer.get_output(0))

        is_pass = is_pass and test_single_layer(tw, output_tensor_list, data, expect_fail_building=True)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_LRN_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.tile(np.array([1, 2, 5], dtype=np.float32).reshape(3, 1, 1), (1, 3, 3)).reshape(1, 3, 3, 3)}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_lrn(tensor, 3, 1.0, 1.0, 0.0001)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_matrix_multiply_layer():
    n_test_case = 4
    is_pass = True
    # Common data
    data = np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5)
    data = {"tensor": data}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            weight_shape = data["tensor"].transpose(0, 1, 3, 2).shape
            layer_weight = tw.network.add_constant(weight_shape, trt.Weights(np.ascontiguousarray(np.ones(weight_shape, dtype=np.float32))))
            layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.NONE)
        elif i == 1:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            weight_shape = data["tensor"].shape  # No transpose compared with `case_simple`
            layer_weight = tw.network.add_constant(weight_shape, trt.Weights(np.ascontiguousarray(np.ones(weight_shape, dtype=np.float32))))
            layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.TRANSPOSE)
        elif i == 2:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            weight_shape = data["tensor"].transpose(0, 1, 3, 2).shape[:-1]  # One less dimension compared with `case_simple`
            layer_weight = tw.network.add_constant(weight_shape, trt.Weights(np.ascontiguousarray(np.ones(weight_shape, dtype=np.float32))))
            layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.VECTOR)
        elif i == 3:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            weight_shape = (1, 1) + data["tensor"].transpose(0, 1, 3, 2).shape[-2:]  # [1,1,5,4]
            layer_weight = tw.network.add_constant(weight_shape, trt.Weights(np.ascontiguousarray(np.ones(weight_shape, dtype=np.float32))))
            layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.NONE)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_NMS_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.random.rand(60).astype(np.float32).reshape(5, 3, 4), "tensor1": np.random.rand(150).astype(np.float32).reshape(5, 3, 10)}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            tw.config.add_optimization_profile(tw.profile)
            layer_max_output = tw.network.add_constant([], np.int32(20).reshape(-1))
            layer = tw.network.add_nms(tensor0, tensor1, layer_max_output.get_output(0))
            layer.topk_box_limit = 100
            layer.bounding_box_format = trt.BoundingBoxFormat.CENTER_SIZES

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0), layer.get_output(1)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_nonzero_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = np.zeros([3, 4, 5]).astype(np.float32)
    data[0, 0, 1] = 1
    data[0, 2, 3] = 2
    data[0, 3, 4] = 3
    data[1, 1, 0] = 4
    data[1, 1, 1] = 5
    data[1, 1, 2] = 6
    data[1, 1, 3] = 7
    data[1, 1, 4] = 8
    data[2, 0, 1] = 9
    data[2, 1, 1] = 10
    data[2, 2, 1] = 11
    data[2, 3, 1] = 12
    data = {"tensor": data}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_non_zero(tensor)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_normalization_layer():
    n_test_case = 3
    is_pass = True
    # Common data
    data0 = np.arange(15, dtype=np.float32).reshape(1, 1, 3, 5)
    data1 = 100 - data0
    data2 = np.ones([1, 1, 3, 5], dtype=np.float32)
    data3 = -data2
    data = {"tensor": np.concatenate([data0, data1, data2, data3], axis=1)}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            shape_scale_bias = (1, 1) + data["tensor"].shape[2:]  # [1, 1, 3, 5]
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer1 = tw.network.add_constant(shape_scale_bias, trt.Weights(np.ones(shape_scale_bias, dtype=np.float32)))
            layer2 = tw.network.add_constant(shape_scale_bias, trt.Weights(np.zeros(shape_scale_bias, dtype=np.float32)))
            layer = tw.network.add_normalization(tensor, layer1.get_output(0), layer2.get_output(0), 1 << 2 | 1 << 3)
            layer.compute_precision = trt.float16
            layer.epsilon = 1e-5
        elif i == 1:
            n_group = 2
            shape_scale_bias = [1, n_group, 1, 1]  # [1, 2, 1, 1]
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer1 = tw.network.add_constant(shape_scale_bias, trt.Weights(np.ones(shape_scale_bias, dtype=np.float32)))
            layer2 = tw.network.add_constant(shape_scale_bias, trt.Weights(np.zeros(shape_scale_bias, dtype=np.float32)))
            layer = tw.network.add_normalization(tensor, layer1.get_output(0), layer2.get_output(0), 1 << 2 | 1 << 3)
            layer.num_groups = n_group  # [Optional] Modify the number of groups
        elif i == 2:
            shape_scale_bias = (1, ) + data["tensor"].shape[1:2] + (1, 1)  # [1, 4, 1, 1]
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer1 = tw.network.add_constant(shape_scale_bias, trt.Weights(np.ones(shape_scale_bias, dtype=np.float32)))
            layer2 = tw.network.add_constant(shape_scale_bias, trt.Weights(np.zeros(shape_scale_bias, dtype=np.float32)))
            layer = tw.network.add_normalization(tensor, layer1.get_output(0), layer2.get_output(0), 1 << 2 | 1 << 3)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_onehot_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.array([[0, 1, 2, 3], [5, 4, 3, 2], [5, 7, 9, 11]], dtype=np.int32)}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            value = tw.network.add_constant([2], np.ascontiguousarray([0, 1], dtype=np.float32))  # [offValue, onValue]
            depth = tw.network.add_constant([], np.ascontiguousarray(16, dtype=np.int32))  # Width of the embedding table, MUST be buildtime constant tensor
            layer = tw.network.add_one_hot(tensor, value.get_output(0), depth.get_output(0), 1)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_padding_layer():
    n_test_case = 2
    is_pass = True
    # Common data
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5) + 1}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_padding_nd(tensor, [1, 2], [3, 4])
        elif i == 1:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_padding_nd(tensor, [-1, 0], [0, -2])

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_ParametricReLU_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5) - 30}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer1 = tw.network.add_constant([1, 1, 1], np.array([0.5], dtype=np.float32))
            layer = tw.network.add_parametric_relu(tensor, layer1.get_output(0))

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_pooling_layer():
    n_test_case = 3
    is_pass = True
    # Common data
    n_hk, n_wk = 2, 2
    data = {"tensor": np.tile(np.arange(9, dtype=np.float32).reshape(3, 3), [1, 1, 2, 3]) + 1}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_pooling_nd(tensor, trt.PoolingType.AVERAGE, [n_hk, n_wk])
        elif i == 1:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_pooling_nd(tensor, trt.PoolingType.MAX_AVERAGE_BLEND, [n_hk, n_wk])
            layer.blend_factor = 0.5
        elif i == 2:
            n_ck = 2
            data1 = np.tile(data["tensor"], (2, 1, 1)).reshape([1, 1, 2, 6, 9])
            data1[0, 0, 1] *= 10
            data1 = {"tensor": data1}
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
            layer = tw.network.add_pooling_nd(tensor, trt.PoolingType.MAX, [n_ck, n_hk, n_wk])

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_ragged_softmax_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    shape = [1, 3, 4, 5]
    data0 = np.ones(np.prod(shape), dtype=np.float32).reshape(shape[1:])
    data1 = np.tile(2 * np.arange(shape[2], dtype=np.int32), (shape[1], 1)).reshape(shape[1], shape[2], 1)
    data = {"tensor": data0, "tensor1": data1}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_ragged_softmax(tensor0, tensor1)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_reduce_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = np.ones([3, 4, 5], dtype=np.float32)
    data = {"tensor": data}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_reduce(tensor, trt.ReduceOperation.SUM, 1 << 1, False)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_resize_layer():
    n_test_case = 7
    is_pass = True
    # Common data
    shape_input = 1, 3, 4, 5
    shape_output = 2, 3, 6, 10
    data = np.arange(np.prod(shape_input), dtype=np.float32).reshape(shape_input)
    data = {"tensor": data}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_resize(tensor)
            layer.shape = shape_output
            input_data = data
        elif i == 1:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_resize(tensor)
            layer.scales = np.array(shape_output) / np.array(shape_input)
            input_data = data
        elif i == 2:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            constant_layer = tw.network.add_constant([4], np.array(shape_output, dtype=np.int32))
            layer = tw.network.add_resize(tensor)
            layer.set_input(1, constant_layer.get_output(0))
            input_data = data
        elif i == 3:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_resize(tensor)
            layer.resize_mode = trt.InterpolationMode.CUBIC
            layer.cubic_coeff = 0.5
            input_data = data
        elif i == 4:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_resize(tensor)
            layer.shape = [shape_input[0], shape_output[1], 1, 1]
            layer.resize_mode = trt.InterpolationMode.LINEAR
            layer.selector_for_single_pixel = trt.ResizeSelector.UPPER
            layer.nearest_rounding = trt.ResizeRoundMode.CEIL
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS
            input_data = data
        elif i == 5:
            data1 = {"tensor": data["tensor"], "tensor1": np.array(shape_output, dtype=np.int32)}
            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), data1["tensor1"].shape)
            tw.profile.set_shape_input(tensor1.name, [1 for _ in shape_input], shape_output, shape_output)
            tw.config.add_optimization_profile(tw.profile)
            layer = tw.network.add_resize(tensor0)
            layer.set_input(1, tensor1)
            input_data = data1
        elif i == 6:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_resize(tensor)
            layer.shape = shape_output
            layer.exclude_outside = 1
            input_data = data

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], input_data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_reverse_sequence_layer():
    n_test_case = 2
    is_pass = True
    # Common data
    shape = [3, 4, 5]
    data = np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1) * 100 + \
        np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1) * 10 + \
        np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2])
    data = {"tensor": data}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            data["tensor1"] = np.array([4, 3, 2, 1], dtype=np.int32)
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_reverse_sequence(tensor, tensor1)
        elif i == 1:
            data["tensor1"] = np.array([3, 2, 1], dtype=np.int32)
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_reverse_sequence(tensor, tensor1)
            layer.batch_axis = 0
            layer.sequence_axis = 1

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_scale_layer():
    n_test_case = 4
    is_pass = True
    # Common data
    shape = [1, 3, 3, 3]
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
    data = {"tensor": data}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            scale = np.ascontiguousarray(np.array([0.5], dtype=np.float32))
            shift = np.ascontiguousarray(np.array([-7.0], dtype=np.float32))
            power = np.ascontiguousarray(np.array([1.0], dtype=np.float32))
            layer = tw.network.add_scale(tensor, trt.ScaleMode.UNIFORM, trt.Weights(shift), trt.Weights(scale), trt.Weights(power))
        elif i == 1:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            shift = np.ascontiguousarray(np.array([-2.5, -7.0, -11.5], dtype=np.float32))
            scale = np.ascontiguousarray(np.array([0.5, 0.5, 0.5], dtype=np.float32))
            power = np.ascontiguousarray(np.array([1, 1, 1], dtype=np.float32))
            layer = tw.network.add_scale(tensor, trt.ScaleMode.CHANNEL, trt.Weights(shift), trt.Weights(scale), trt.Weights(power))
        elif i == 2:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            shift = np.ascontiguousarray(np.full(shape[1:], -7.0, dtype=np.float32))
            scale = np.ascontiguousarray(np.full(shape[1:], 0.5, dtype=np.float32))
            power = np.ascontiguousarray(np.ones(shape[1:], dtype=np.float32))
            layer = tw.network.add_scale(tensor, trt.ScaleMode.ELEMENTWISE, trt.Weights(shift), trt.Weights(scale), trt.Weights(power))
        elif i == 3:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            shift = np.ascontiguousarray(np.array([-2.5, -7.0, -11.5], dtype=np.float32))
            scale = np.ascontiguousarray(np.array([0.5, 0.5, 0.5], dtype=np.float32))
            power = np.ascontiguousarray(np.array([1, 1, 1], dtype=np.float32))
            layer = tw.network.add_scale_nd(tensor, trt.ScaleMode.CHANNEL, trt.Weights(shift), trt.Weights(scale), trt.Weights(power), 0)
            layer.channel_axis = 1

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_scatter_layer():
    n_test_case = 3
    is_pass = True
    # Common data

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            shape = 1, 3, 4, 5
            data0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            data1 = np.tile(np.arange(shape[2], dtype=np.int32), [shape[0], shape[1], 1, shape[3]]).reshape(shape)
            data2 = -data0
            data = {"tensor": data0, "tensor1": data1, "tensor2": data2}
            scatter_axis = 2

            def scatter_cpu_element(data0, data1, data2, axis):
                nB, nC, nH, nW = data0.shape
                output = data0
                for n in range(nB):
                    for c in range(nC):
                        for h in range(nH):
                            for w in range(nW):
                                if axis == 0:
                                    output[data1[n, c, h, w], c, h, w] = data2[n, c, h, w]
                                if axis == 1:
                                    output[n, data1[n, c, h, w], h, w] = data2[n, c, h, w]
                                if axis == 2:
                                    output[n, c, data1[n, c, h, w], w] = data2[n, c, h, w]
                                if axis == 3:
                                    output[n, c, h, data1[n, c, h, w]] = data2[n, c, h, w]
                                else:
                                    print("Fail scattering at axis %d " % axis)
                                    return None
                return output

            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
            layer = tw.network.add_scatter(tensor0, tensor1, tensor2, trt.ScatterMode.ELEMENT)
            layer.axis = scatter_axis
        elif i == 1:
            shape = [2, 3, 4, 5]
            data0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            data1 = np.array([[[0, 2, 1, 1], [1, 0, 3, 2], [0, 1, 2, 3]], [[1, 2, 1, 1], [0, 0, 3, 2], [1, 1, 2, 3]]], dtype=np.int32)
            data2 = -np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape[0], shape[1])
            data = {"tensor": data0, "tensor1": data1, "tensor2": data2}

            def scatter_cpu_nd(data0, data1, data2):
                output = data0
                for i in range(data1.shape[0]):
                    for j in range(data1.shape[1]):
                        #print(f"{i=},{j=},index={data1[i,j]},updateValue={data2[i, j]}")
                        output[data1[i, j][0], data1[i, j][1], data1[i, j][2], data1[i, j][3]] = data2[i, j]
                return output

            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
            layer = tw.network.add_scatter(tensor0, tensor1, tensor2, trt.ScatterMode.ND)
        elif i == 2:
            shape = [2, 3, 4, 5]
            data0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            data1 = np.array([[0, 2, 1], [1, 0, 3], [0, 1, 2], [1, 2, 1], [0, 0, 3], [1, 1, 2]], dtype=np.int32)
            data2 = -np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
            data = {"tensor": data0, "tensor1": data1, "tensor2": data2}

            def scatter_cpu_nd(data0, data1, data2):
                output = data0
                for i in range(data1.shape[0]):
                    #print(f"{i=},index={data1[i]},updateValue={data2[i]}")
                    output[data1[i][0], data1[i][1], data1[i][2]] = data2[i]
                return output

            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
            layer = tw.network.add_scatter(tensor0, tensor1, tensor2, trt.ScatterMode.ND)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_select_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    shape = [1, 3, 4, 5]
    data0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    data1 = -data0
    data2 = (np.arange(np.prod(shape)) % 2).astype(bool).reshape(shape)
    data = {"tensor": data0, "tensor1": data1, "tensor2": data2}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
            layer = tw.network.add_select(tensor2, tensor0, tensor1)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_shape_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(3, 4, 5)}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_shape(tensor)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_shuffle_layer():
    n_test_case = 6
    is_pass = True
    # Common data
    shape = 1, 3, 4, 5
    data = np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
        np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
        np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
        np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]) * 1
    data = {"tensor": data}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_shuffle(tensor)
            layer.first_transpose = (0, 2, 1, 3)
            layer.reshape_dims = (1, 4, 5, 3)
            layer.second_transpose = (0, 2, 1, 3)
            input_data = data
        elif i == 1:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            one_layer = tw.network.add_constant([1], np.array([1], dtype=np.int64))  # Shape constant tensor need to be INT64
            shape_layer_0 = tw.network.add_shape(tensor)
            shape_layer_1 = tw.network.add_concatenation([shape_layer_0.get_output(0), one_layer.get_output(0)])
            shape_layer_1.axis = 0
            shuffle_layer = tw.network.add_shuffle(tensor)
            shuffle_layer.set_input(1, shape_layer_1.get_output(0))
            shape_layer_2 = tw.network.add_shape(shuffle_layer.get_output(0))
            shape_layer_3 = tw.network.add_slice(shape_layer_2.get_output(0), [0], [4], [1])
            shuffle_layer_2 = tw.network.add_shuffle(shuffle_layer.get_output(0))  # remove the tail dimension 1 to input tensor
            shuffle_layer_2.set_input(1, shape_layer_3.get_output(0))
            layer = shuffle_layer_2
        elif i == 2:
            data1 = {"tensor": data["tensor"], "tensor1": np.array([1, 4, 5, 3], dtype=np.int32)}
            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), data1["tensor1"].shape)
            tw.profile.set_shape_input(tensor1.name, [1 for _ in data1["tensor1"]], data1["tensor1"], data1["tensor1"])
            tw.config.add_optimization_profile(tw.profile)
            layer = tw.network.add_shuffle(tensor0)
            layer.set_input(1, tensor1)
            input_data = data1
        elif i == 3:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_shuffle(tensor)
            layer.reshape_dims = (0, 0, -1)
            input_data = data
        elif i == 4:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_shuffle(tensor)
            layer.zero_is_placeholder = True
            layer.reshape_dims = (0, 0, 0, 0)
            input_data = data
        elif i == 5:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            constantLayer = tw.network.add_constant([0], trt.Weights(trt.float32))
            shuffleLayer = tw.network.add_shuffle(constantLayer.get_output(0))
            shuffleLayer.zero_is_placeholder = False
            shuffleLayer.reshape_dims = (1, 3, 4, 0)
            layer = tw.network.add_concatenation([tensor, shuffleLayer.get_output(0)])
            layer.axis = 3
            input_data = data

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], input_data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_slice_layer():
    n_test_case = 5
    is_pass = True
    # Common data
    shape = [1, 3, 4, 5]
    data = np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
        np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
        np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
        np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3])
    data = {"tensor": data}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [1, 2, 3, 4], [1, 1, 1, 1])
            input_data = data
        elif i == 1:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer1 = tw.network.add_constant([1], np.array([-1], dtype=np.float32))  # Value of out-of-bound
            layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [1, 2, 3, 4], [1, 2, 2, 2])
            layer.mode = trt.SampleMode.FILL
            layer.set_input(4, layer1.get_output(0))
            input_data = data
        elif i == 2:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer1 = tw.network.add_constant([4], np.array([0, 0, 0, 0], dtype=np.int32))
            layer2 = tw.network.add_constant([4], np.array([1, 2, 3, 4], dtype=np.int32))
            layer3 = tw.network.add_constant([4], np.array([1, 1, 1, 1], dtype=np.int32))
            layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
            layer.set_input(1, layer1.get_output(0))
            layer.set_input(2, layer2.get_output(0))
            layer.set_input(3, layer3.get_output(0))
            input_data = data
        elif i == 3:
            data1 = {"tensor": data["tensor"]}
            data1["tensor1"] = np.array([0, 0, 0, 0], dtype=np.int32)
            data1["tensor2"] = np.array([1, 2, 3, 4], dtype=np.int32)
            data1["tensor3"] = np.array([1, 1, 1, 1], dtype=np.int32)
            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), data1["tensor1"].shape)
            tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data1["tensor2"].dtype), data1["tensor2"].shape)
            tensor3 = tw.network.add_input("tensor3", datatype_np_to_trt(data1["tensor3"].dtype), data1["tensor3"].shape)
            tw.profile.set_shape_input(tensor1.name, [0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2])
            tw.profile.set_shape_input(tensor2.name, [1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 4, 5])
            tw.profile.set_shape_input(tensor3.name, [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1])
            tw.config.add_optimization_profile(tw.profile)
            layer = tw.network.add_slice(tensor0, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
            layer.set_input(1, tensor1)
            layer.set_input(2, tensor2)
            layer.set_input(3, tensor3)
            input_data = data1
        elif i == 4:
            continue  # Disable this case since TRT  does not support such usage yet
            data1 = {"tensor": data["tensor"]}
            data1["tensor1"] = np.array([1, 2, 3, 4], dtype=np.int32)
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), [-1 for _ in data1["tensor1"].shape])
            tw.profile.set_shape(tensor1.name, data1["tensor1"].shape, data1["tensor1"].shape, data1["tensor1"].shape)
            tw.config.add_optimization_profile(tw.profile)
            layer1 = tw.network.add_elementwise(tensor1, tensor1, trt.ElementWiseOperation.SUM)  # Compute shape tensor from earlier layer
            layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1])
            layer.set_input(2, layer1.get_output(0))
            input_data = data1

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], input_data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_softmax_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.arange(9, dtype=np.float32).reshape(3, 3) - 4}  # [0, 8] -> [-4, 4]}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_softmax(tensor)
            layer.axes = 1 << 1

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_squeeze_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.ones([3, 1, 1, 4, 5], dtype=np.float32)}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer_axis = tw.network.add_constant(shape=[2], weights=np.array([1, 2], dtype=np.int32))
            layer = tw.network.add_squeeze(tensor, layer_axis.get_output(0))

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_topk_layer():
    n_test_case = 3
    is_pass = True
    # Common data
    data = {"tensor": np.random.permutation(np.arange(60, dtype=np.float32)).reshape(3, 4, 5)}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 2, 1 << 1)
            input_data = data
        elif i == 1:
            data1 = {"tensor": data["tensor"], "tensor1": np.array([2], dtype=np.int32)}
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), [])
            tw.profile.set_shape_input(tensor1.name, [1], [2], [3])
            tw.config.add_optimization_profile(tw.profile)
            layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 1, 1 << 1)
            layer.set_input(1, tensor1)
            input_data = data1
        elif i == 2:
            data1 = {"tensor": data["tensor"], "tensor1": np.array([3, -1], dtype=np.int32)}  # tensor1 is a execution input tensor
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), [-1 for _ in data1["tensor1"].shape])
            tw.profile.set_shape(tensor1.name, [1 for _ in data1["tensor1"].shape], data1["tensor1"].shape, data1["tensor1"].shape)
            tw.config.add_optimization_profile(tw.profile)
            layer1 = tw.network.add_reduce(tensor1, trt.ReduceOperation.SUM, 1 << 0, False)  # Compute K from earlier layer
            layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 1, 1 << 1)
            layer.set_input(1, layer1.get_output(0))
            input_data = data1

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], input_data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_unary_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.arange(9, dtype=np.float32).reshape(3, 3) - 4}  # [0, 8] -> [-4, 4]}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_unary(tensor, trt.UnaryOperation.ABS)

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_unsqueeze_layer():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.ones([3, 4, 5], dtype=np.float32)}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer_axis = tw.network.add_constant(shape=[2], weights=np.array([1, 2], dtype=np.int32))
            layer = tw.network.add_unsqueeze(tensor, layer_axis.get_output(0))

        is_pass = is_pass and test_single_layer(tw, [layer.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_if_structure():
    n_test_case = 1
    is_pass = True
    # Common data
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5) + 1}
    data1 = {"tensor": data["tensor"] - 1}

    for i in range(n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            # Extract the scalar first element
            layer1 = tw.network.add_shuffle(tensor)
            layer1.reshape_dims = [-1]
            layer2 = tw.network.add_slice(layer1.get_output(0), [0], [1], [1])
            layer3 = tw.network.add_shuffle(layer2.get_output(0))
            layer3.reshape_dims = []
            layer4 = tw.network.add_identity(layer3.get_output(0))
            layer4.set_output_type(0, trt.bool)
            if_structure = tw.network.add_if_conditional()
            layer_input = if_structure.add_input(tensor)
            if_structure.set_condition(layer4.get_output(0))
            # Branch of condition is true
            layer_true = tw.network.add_elementwise(layer_input.get_output(0), layer_input.get_output(0), trt.ElementWiseOperation.SUM)
            # Branch of condition is false
            layer_false = tw.network.add_identity(layer_input.get_output(0))
            layer_output = if_structure.add_output(layer_true.get_output(0), layer_false.get_output(0))

        is_pass = is_pass and test_single_layer(tw, [layer_output.get_output(0)], data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

@case_mark
def test_loop_structure():
    n_test_case = 5
    is_pass = True
    # Common data
    data = {"tensor": np.ones([1, 2, 3, 4], dtype=np.float32)}

    for i in range(2, n_test_case):
        print(f"Run test case {i}")
        tw = TRTWrapperV2()
        if i == 0:
            t = np.array([5], dtype=np.int32)  # Number of iterations
            v = np.array([6], dtype=np.int32)  # Number of output to keep, we usually use v == t

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            loop = tw.network.add_loop()

            layer_t = tw.network.add_constant((), t)
            loop.add_trip_limit(layer_t.get_output(0), trt.TripLimit.COUNT)

            layer_recurrence = loop.add_recurrence(tensor)
            layer_body = tw.network.add_elementwise(layer_recurrence.get_output(0), layer_recurrence.get_output(0), trt.ElementWiseOperation.SUM)
            layer_recurrence.set_input(1, layer_body.get_output(0))

            layer_output = loop.add_loop_output(layer_recurrence.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
            layer_output1 = loop.add_loop_output(layer_body.get_output(0), trt.LoopOutput.CONCATENATE, 0)
            # Keep output of iteration [1, t] if passing layer_body to loop.add_loop_output
            # Keep output of iteration [0, t-1] if passing layer_recurrence to loop.add_loop_output

            layer_v = tw.network.add_constant((), v)
            layer_output1.set_input(1, layer_v.get_output(0))
            # Output shape on the iteration axis depends on v,
            # The output of first v ierations are kept if v <= t,
            # Or 0 padding is used for the part of v > t.
            output_tensor_list = [layer_output.get_output(0), layer_output1.get_output(0)]
            input_data = data

        elif i == 1:
            data1 = {"tensor": data["tensor"], "tensor1": np.array(5, dtype=np.int32)}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), data1["tensor1"].shape)  # Set number of iteration at runtime
            tw.profile.set_shape_input(tensor1.name, [1], [6], [10])
            tw.config.add_optimization_profile(tw.profile)
            loop = tw.network.add_loop()

            loop.add_trip_limit(tensor1, trt.TripLimit.COUNT)

            layer_recurrence = loop.add_recurrence(tensor)
            layer_body = tw.network.add_elementwise(layer_recurrence.get_output(0), layer_recurrence.get_output(0), trt.ElementWiseOperation.SUM)
            layer_recurrence.set_input(1, layer_body.get_output(0))

            layer_output = loop.add_loop_output(layer_recurrence.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
            layer_output1 = loop.add_loop_output(layer_body.get_output(0), trt.LoopOutput.CONCATENATE, 0)
            layer_output1.set_input(1, tensor1)
            output_tensor_list = [layer_output.get_output(0), layer_output1.get_output(0)]
            input_data = data1

        elif i == 2:
            threshold = np.array([32], dtype=np.float32)
            v = np.array([6], dtype=np.int32)  # Number of output to keep, we usually use v == t

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)

            loop = tw.network.add_loop()
            layer_recurrence = loop.add_recurrence(tensor)
            layer_threshold = tw.network.add_constant((), threshold)

            # Extract the scalar first element of `layer_recurrence`
            layer1 = tw.network.add_shuffle(layer_recurrence.get_output(0))
            layer1.reshape_dims = [-1]
            layer2 = tw.network.add_slice(layer1.get_output(0), [0], [1], [1])
            layer3 = tw.network.add_shuffle(layer2.get_output(0))
            layer3.reshape_dims = []

            # Compare the element with threshold
            layer4 = tw.network.add_elementwise(layer_threshold.get_output(0), layer3.get_output(0), trt.ElementWiseOperation.SUB)
            layer5 = tw.network.add_activation(layer4.get_output(0), trt.ActivationType.RELU)
            layer6 = tw.network.add_identity(layer5.get_output(0))
            layer6.set_output_type(0, trt.bool)
            layer6.get_output(0).dtype = trt.bool

            loop.add_trip_limit(layer6.get_output(0), trt.TripLimit.WHILE)

            layer_body = tw.network.add_elementwise(layer_recurrence.get_output(0), layer_recurrence.get_output(0), trt.ElementWiseOperation.SUM)
            layer_recurrence.set_input(1, layer_body.get_output(0))

            layer_output = loop.add_loop_output(layer_recurrence.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
            layer_output1 = loop.add_loop_output(layer_recurrence.get_output(0), trt.LoopOutput.CONCATENATE, 0)
            layer_v = tw.network.add_constant((), v)
            layer_output1.set_input(1, layer_v.get_output(0))

            output_tensor_list = [layer_output.get_output(0), layer_output1.get_output(0)]
            input_data = data

        elif i == 3:
            _, n_c, n_h, n_w = data["tensor"].shape

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)

            loop = tw.network.add_loop()
            iterator = loop.add_iterator(tensor, 1, False)  # Build a iterator with tensor, axis and weight_hether to reverse
            print(f"{iterator.reverse=}")  # Read-only attribution

            layer_t = tw.network.add_constant((), np.array([n_c], dtype=np.int32))
            loop.add_trip_limit(layer_t.get_output(0), trt.TripLimit.COUNT)

            layer_initial = tw.network.add_constant([1, n_h, n_w], np.ones(n_h * n_w, dtype=np.float32))
            rLayer = loop.add_recurrence(layer_initial.get_output(0))

            layer_body = tw.network.add_elementwise(rLayer.get_output(0), iterator.get_output(0), trt.ElementWiseOperation.SUM)
            rLayer.set_input(1, layer_body.get_output(0))

            layer_output = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
            layer_output1 = loop.add_loop_output(layer_body.get_output(0), trt.LoopOutput.CONCATENATE, 0)
            layer_output1.set_input(1, layer_t.get_output(0))

            output_tensor_list = [layer_output.get_output(0), layer_output1.get_output(0)]
            input_data = data

        elif i == 4:
            n_b, n_isl, n_ih, n_h = 3, 4, 7, 5  # batch_size, input_sequence_length, input_hidden_size, hidden_size
            x = np.ones([n_b, n_isl, n_ih], dtype=np.float32)
            h0 = np.ones([n_b, n_h], dtype=np.float32)  # Initial hidden state
            c0 = np.zeros([n_b, n_h], dtype=np.float32)  # Initial cell state
            data1 = {"x": x, "h0": h0, "c0": c0}

            weight_x = np.ones((n_h, n_ih), dtype=np.float32)  # Weight of X->H, we use the same weight for each gate in this example
            weight_h = np.ones((n_h, n_h), dtype=np.float32)  # Weight of H->H
            bias_x = np.zeros(n_h, dtype=np.float32)  # Bias of X->H
            bias_h = np.zeros(n_h, dtype=np.float32)  # Bias of H->H

            input_x = tw.network.add_input("x", datatype_np_to_trt(data1["x"].dtype), [-1, -1, n_ih])
            input_h0 = tw.network.add_input("h0", datatype_np_to_trt(data1["h0"].dtype), [-1, n_h])
            input_c0 = tw.network.add_input("c0", datatype_np_to_trt(data1["c0"].dtype), [-1, n_h])
            tw.profile.set_shape(input_x.name, [1, 1, n_ih], [n_b, n_isl, n_ih], [n_b, n_isl * 2, n_ih])
            tw.profile.set_shape(input_h0.name, [1, n_h], [n_b, n_h], [n_b, n_h])
            tw.profile.set_shape(input_c0.name, [1, n_h], [n_b, n_h], [n_b, n_h])
            tw.config.add_optimization_profile(tw.profile)

            def gate(tensor_x, weight_x, tensor_h, weight_h, bias, b_sigmoid):
                layer_h0 = tw.network.add_matrix_multiply(tensor_x, trt.MatrixOperation.NONE, weight_x, trt.MatrixOperation.NONE)
                layer_h1 = tw.network.add_matrix_multiply(tensor_h, trt.MatrixOperation.NONE, weight_h, trt.MatrixOperation.NONE)
                layer_h2 = tw.network.add_elementwise(layer_h0.get_output(0), layer_h1.get_output(0), trt.ElementWiseOperation.SUM)
                layer_h3 = tw.network.add_elementwise(layer_h2.get_output(0), bias, trt.ElementWiseOperation.SUM)
                layer_h4 = tw.network.add_activation(layer_h3.get_output(0), trt.ActivationType.SIGMOID if b_sigmoid else trt.ActivationType.TANH)
                return layer_h4

            loop = tw.network.add_loop()

            layer_t0 = tw.network.add_shape(input_x)
            layer_t1 = tw.network.add_slice(layer_t0.get_output(0), [1], [1], [1])
            layer_t2 = tw.network.add_shuffle(layer_t1.get_output(0))
            layer_t2.reshape_dims = ()
            layer_t3 = tw.network.add_cast(layer_t2.get_output(0), trt.DataType.INT32)
            loop.add_trip_limit(layer_t3.get_output(0), trt.TripLimit.COUNT)

            iterator = loop.add_iterator(input_x, 1, False)  # Get a slice [n_b, n_ih] from input_x in each iteration
            tensor_x = iterator.get_output(0)
            layer_hidden_h = loop.add_recurrence(input_h0)  # Initial hidden state and cell state. There are multiple loop variables in a loop.
            layer_hidden_c = loop.add_recurrence(input_c0)

            layer_weight_x = tw.network.add_constant([n_ih, n_h], trt.Weights(np.ascontiguousarray(weight_x.transpose())))
            layer_weight_h = tw.network.add_constant([n_h, n_h], trt.Weights(np.ascontiguousarray(weight_h.transpose())))
            layer_bias = tw.network.add_constant([1, n_h], trt.Weights(np.ascontiguousarray(bias_x + bias_h)))

            weight_x = layer_weight_x.get_output(0)
            tensor_h = layer_hidden_h.get_output(0)
            weight_h = layer_weight_h.get_output(0)
            bias = layer_bias.get_output(0)
            gate_i = gate(tensor_x, weight_x, tensor_h, weight_h, bias, True)
            gate_f = gate(tensor_x, weight_x, tensor_h, weight_h, bias, True)
            gate_c = gate(tensor_x, weight_x, tensor_h, weight_h, bias, False)
            gate_o = gate(tensor_x, weight_x, tensor_h, weight_h, bias, True)

            layer_body = tw.network.add_elementwise(gate_f.get_output(0), layer_hidden_c.get_output(0), trt.ElementWiseOperation.PROD)
            layer_body1 = tw.network.add_elementwise(gate_i.get_output(0), gate_c.get_output(0), trt.ElementWiseOperation.PROD)
            layer_hidden_c1 = tw.network.add_elementwise(layer_body.get_output(0), layer_body1.get_output(0), trt.ElementWiseOperation.SUM)
            layer_body2 = tw.network.add_activation(layer_hidden_c1.get_output(0), trt.ActivationType.TANH)
            layer_hidden_h1 = tw.network.add_elementwise(gate_o.get_output(0), layer_body2.get_output(0), trt.ElementWiseOperation.PROD)

            layer_hidden_h.set_input(1, layer_hidden_h1.get_output(0))
            layer_hidden_c.set_input(1, layer_hidden_c1.get_output(0))

            layer_output = loop.add_loop_output(layer_hidden_h.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
            layer_output.get_output(0).name = "y"
            layer_output1 = loop.add_loop_output(layer_hidden_h1.get_output(0), trt.LoopOutput.CONCATENATE, 1)
            layer_output1.get_output(0).name = "h1"
            layer_output1.set_input(1, layer_t2.get_output(0))
            layer_output2 = loop.add_loop_output(layer_hidden_c.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
            layer_output2.get_output(0).name = "c1"

            output_tensor_list = [layer_output.get_output(0), layer_output1.get_output(0), layer_output2.get_output(0)]
            input_data = data1

        is_pass = is_pass and test_single_layer(tw, output_tensor_list, input_data)
    print(f"{n_test_case:2d} case pass: {is_pass}")
    return is_pass

if __name__ == "__main__":
    all_pass = True

    # all_pass = all_pass and test_activation_layer()
    # all_pass = all_pass and test_assert_layer()
    # all_pass = all_pass and test_cast_layer()
    # all_pass = all_pass and test_concatenation_layer()
    # all_pass = all_pass and test_constant_layer()
    # all_pass = all_pass and test_convolution_layer()
    # all_pass = all_pass and test_cumulative_layer()
    # all_pass = all_pass and test_deconvolution_layer()
    # all_pass = all_pass and test_dynamic_quantize_layer()
    # all_pass = all_pass and test_einsum_layer()
    # all_pass = all_pass and test_elementwise_layer()
    # all_pass = all_pass and test_fill_layer()
    # all_pass = all_pass and test_gather_layer()
    # all_pass = all_pass and test_grid_sample_layer()
    # all_pass = all_pass and test_identity_layer()
    # all_pass = all_pass and test_if_structure()
    # all_pass = all_pass and test_LRN_layer()
    # all_pass = all_pass and test_matrix_multiply_layer()
    # all_pass = all_pass and test_NMS_layer()
    # all_pass = all_pass and test_nonzero_layer()
    # all_pass = all_pass and test_normalization_layer()
    # all_pass = all_pass and test_onehot_layer()
    # all_pass = all_pass and test_padding_layer()
    # all_pass = all_pass and test_ParametricReLU_layer()
    # all_pass = all_pass and test_pooling_layer()
    # all_pass = all_pass and test_ragged_softmax_layer()
    # all_pass = all_pass and test_reduce_layer()
    # all_pass = all_pass and test_resize_layer()
    # all_pass = all_pass and test_reverse_sequence_layer()
    # all_pass = all_pass and test_scale_layer()
    # all_pass = all_pass and test_scatter_layer()
    # all_pass = all_pass and test_select_layer()
    # all_pass = all_pass and test_shape_layer()
    # all_pass = all_pass and test_shuffle_layer()
    # all_pass = all_pass and test_slice_layer()
    # all_pass = all_pass and test_softmax_layer()
    # all_pass = all_pass and test_squeeze_layer()
    # all_pass = all_pass and test_topk_layer()
    # all_pass = all_pass and test_unary_layer()
    # all_pass = all_pass and test_unsqueeze_layer()

    # all_pass = all_pass and test_qdq_structure()
    all_pass = all_pass and test_loop_structure()

    print(f"All test pass: {all_pass}")
    print("Finish")
