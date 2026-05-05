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

from collections import OrderedDict

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import (CookbookGpuAllocator, CookbookGpuAsyncAllocator, TRTWrapperV1, case_mark, load_mnist_network_trt, load_plugin_files, get_plugin)

shape = [1, 1, 28, 28]
data = {"x": np.random.rand(np.prod(shape)).astype(np.float32).reshape(shape) * 2 - 1}
stress_shape = [1, 512, 512]
stress_data = {"x": np.random.rand(np.prod(stress_shape)).astype(np.float32).reshape(stress_shape) * 2 - 1}

plugin_file_list = ["./NothingPlugin.so"]

def run_with_gpu_allocator(tw, input_data, gpu_allocator):
    # Work similar as TRTWrapperV1.setup()
    tw.runtime = trt.Runtime(tw.logger)
    tw.runtime.gpu_allocator = gpu_allocator  # can be assign GPU Allocator to Runtime or ExecutionContext

    # Work similar as TRTWrapperV1.setup()
    tw.engine = tw.runtime.deserialize_cuda_engine(tw.engine_bytes)

    tw.tensor_name_list = [tw.engine.get_tensor_name(i) for i in range(tw.engine.num_io_tensors)]
    tw.n_input = sum([tw.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in tw.tensor_name_list])
    tw.n_output = tw.engine.num_io_tensors - tw.n_input

    tw.context = tw.engine.create_execution_context()
    # tw.context.temporary_allocator = gpu_allocator  # can be assign GPU Allocator to Runtime or ExecutionContext

    for name, value in input_data.items():
        tw.context.set_input_shape(name, value.shape)

    # Print information of input / output tensors
    for name in tw.tensor_name_list:
        mode = tw.engine.get_tensor_mode(name)
        data_type = tw.engine.get_tensor_dtype(name)
        buildtime_shape = tw.engine.get_tensor_shape(name)
        runtime_shape = tw.context.get_tensor_shape(name)
        print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {runtime_shape}, {name}")

    tw.buffer = OrderedDict()
    for name in tw.tensor_name_list:
        data_type = tw.engine.get_tensor_dtype(name)
        runtime_shape = tw.context.get_tensor_shape(name)
        n_byte = trt.volume(runtime_shape) * data_type.itemsize
        host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
        device_buffer = cudart.cudaMalloc(n_byte)[1]
        tw.buffer[name] = [host_buffer, device_buffer, n_byte]

    for name, value in input_data.items():
        tw.buffer[name][0] = np.ascontiguousarray(value)

    for name in tw.tensor_name_list:
        tw.context.set_tensor_address(name, tw.buffer[name][1])

    tw.infer(b_print_io=False)

@case_mark
def case_mnist():
    tw = TRTWrapperV1()

    load_mnist_network_trt(tw)
    tw.build()

    allocator = CookbookGpuAllocator(log=True)
    run_with_gpu_allocator(tw, data, allocator)

    print("After inference")  # See what happens after all inference work is done.
    return

@case_mark
def case_normal(index: int = 0):
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("x", trt.float32, [-1, stress_shape[1], stress_shape[2]])
    tw.profile.set_shape(tensor.name, [1, stress_shape[1], stress_shape[2]], [1, stress_shape[1], stress_shape[2]], [4, stress_shape[1], stress_shape[2]])

    if index == 0:
        print("Identity Layer")
        _0 = tw.network.add_identity(tensor)
        _0 = _0.get_output(0)

    elif index == 1:
        print("GEMM + ReLU")
        _0 = tensor
        for i in range(1):
            w = np.random.rand(1, 512, 512).astype(np.float32)
            np.random.rand(1, 1, 512).astype(np.float32)
            _1 = tw.network.add_constant(w.shape, trt.Weights(np.ascontiguousarray(w)))
            _2 = tw.network.add_matrix_multiply(_0, trt.MatrixOperation.NONE, _1.get_output(0), trt.MatrixOperation.NONE)
            _3 = tw.network.add_activation(_2.get_output(0), trt.ActivationType.RELU)
            _0 = _3.get_output(0)

    elif index == 2:
        print("( GEMM + ReLU ) * 10")
        _0 = tensor
        for i in range(10):
            w = np.random.rand(1, 512, 512).astype(np.float32)
            np.random.rand(1, 1, 512).astype(np.float32)
            _1 = tw.network.add_constant(w.shape, trt.Weights(np.ascontiguousarray(w)))
            _2 = tw.network.add_matrix_multiply(_0, trt.MatrixOperation.NONE, _1.get_output(0), trt.MatrixOperation.NONE)
            _3 = tw.network.add_activation(_2.get_output(0), trt.ActivationType.RELU)
            _0 = _3.get_output(0)

    elif index == 3:
        print("sin(x) + x^T + sum(x,-1)")
        _0 = tensor
        for i in range(1):
            _1 = tw.network.add_unary(_0, trt.UnaryOperation.SIN)
            _2 = tw.network.add_shuffle(_0)
            _2.first_transpose = (0, 2, 1)
            _3 = tw.network.add_reduce(_0, trt.ReduceOperation.SUM, 1 << 2, True)
            _4 = tw.network.add_elementwise(_1.get_output(0), _2.get_output(0), trt.ElementWiseOperation.SUM)
            _5 = tw.network.add_elementwise(_4.get_output(0), _3.get_output(0), trt.ElementWiseOperation.SUM)
            _0 = _5.get_output(0)

    elif index == 4:
        print("( sin(x) + x^T + sum(x,-1) ) * 10")
        _0 = tensor
        for i in range(10):
            _1 = tw.network.add_unary(_0, trt.UnaryOperation.SIN)
            _2 = tw.network.add_shuffle(_0)
            _2.first_transpose = (0, 2, 1)
            _3 = tw.network.add_reduce(_0, trt.ReduceOperation.SUM, 1 << 2, True)
            _4 = tw.network.add_elementwise(_1.get_output(0), _2.get_output(0), trt.ElementWiseOperation.SUM)
            _5 = tw.network.add_elementwise(_4.get_output(0), _3.get_output(0), trt.ElementWiseOperation.SUM)
            _0 = _5.get_output(0)

    elif index == 5:
        print("( GEMM + ReLU ) * 10 + UNet")
        tensor_list = []
        _0 = tensor
        for i in range(5):
            w = np.random.rand(1, 512, 512).astype(np.float32)
            np.random.rand(1, 1, 512).astype(np.float32)
            _1 = tw.network.add_constant(w.shape, trt.Weights(np.ascontiguousarray(w)))
            _2 = tw.network.add_matrix_multiply(_0, trt.MatrixOperation.NONE, _1.get_output(0), trt.MatrixOperation.NONE)
            _3 = tw.network.add_activation(_2.get_output(0), trt.ActivationType.RELU)
            _0 = _3.get_output(0)
            tensor_list = tensor_list + [_0]
        for i in range(5):
            w = np.random.rand(1, 512, 512).astype(np.float32)
            np.random.rand(1, 1, 512).astype(np.float32)
            _1 = tw.network.add_constant(w.shape, trt.Weights(np.ascontiguousarray(w)))
            _2 = tw.network.add_matrix_multiply(_0, trt.MatrixOperation.NONE, _1.get_output(0), trt.MatrixOperation.NONE)
            _3 = tw.network.add_activation(_2.get_output(0), trt.ActivationType.RELU)
            _4 = tensor_list.pop()
            _5 = tw.network.add_elementwise(_3.get_output(0), _4, trt.ElementWiseOperation.SUM)
            _0 = _5.get_output(0)

    elif index == 6:
        print("Plugin layer with 0B workspace")
        load_plugin_files(plugin_file_list, tw.logger)
        plugin_info_dict = {
            "NothingPluginLayer": dict(
                name="Nothing",
                version="1",
                namespace="",
                argument_dict=dict(size=np.array([0], dtype=np.float32)),
                number_input_tensor=1,
                number_input_shape_tensor=0,
                plugin_api_version="3",
            )
        }
        _0 = tw.network.add_plugin_v3([tensor], [], get_plugin(plugin_info_dict["NothingPluginLayer"]))
        _0 = _0.get_output(0)

    elif index == 7:
        print("Plugin layer with 1GiB workspace")
        load_plugin_files(plugin_file_list, tw.logger)
        plugin_info_dict = {
            "NothingPluginLayer": dict(
                name="Nothing",
                version="1",
                namespace="",
                argument_dict=dict(size=np.array([1024 ** 3], dtype=np.float32)),
                number_input_tensor=1,
                number_input_shape_tensor=0,
                plugin_api_version="3",
            )
        }
        _0 = tw.network.add_plugin_v3([tensor], [], get_plugin(plugin_info_dict["NothingPluginLayer"]))
        _0 = _0.get_output(0)

    elif index == 8:
        print("Plugin layer with 1 GiB workspace * 10")
        load_plugin_files(plugin_file_list, tw.logger)
        plugin_info_dict = {
            "NothingPluginLayer": dict(
                name="Nothing",
                version="1",
                namespace="",
                argument_dict=dict(size=np.array([1024 ** 3], dtype=np.float32)),
                number_input_tensor=1,
                number_input_shape_tensor=0,
                plugin_api_version="3",
            )
        }
        _0 = tensor
        for i in range(10):
            _0 = tw.network.add_plugin_v3([_0], [], get_plugin(plugin_info_dict["NothingPluginLayer"]))
            _0 = _0.get_output(0)

    tw.build([_0])

    allocator = CookbookGpuAllocator(log=True)
    run_with_gpu_allocator(tw, stress_data, allocator)

    print("After inference")
    return

@case_mark
def case_gpu_async_allocator():

    tw = TRTWrapperV1()

    load_mnist_network_trt(tw)
    tw.build()

    tw.runtime = trt.Runtime(tw.logger)
    tw.runtime.gpu_allocator = CookbookGpuAsyncAllocator()

    tw.setup(data)

    tw.infer(b_print_io=False)

    print("After inference")
    return

if __name__ == "__main__":

    case_mnist()

    for index in range(9):
        case_normal(index)

    case_gpu_async_allocator()

    print("Finish")
