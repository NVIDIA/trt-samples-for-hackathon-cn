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

import ctypes
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1, build_mnist_network_trt, case_mark

input_data_file = Path("/trtcookbook/00-Data/data/InferenceData.npy")
input_data = {"x": np.load(input_data_file)}

@case_mark
def case_pinned_memory_api():
    data = np.zeros([2, 3], dtype=np.float32)
    n_element = np.prod(data.shape)

    # Allocate pinned memory
    _, pBuffer = cudart.cudaHostAlloc(data.nbytes, cudart.cudaHostAllocWriteCombined)
    # Map numpy array to the pinned memory (similar as "copy data from numpy array to pinned memory")
    pointer = ctypes.cast(pBuffer, ctypes.POINTER(ctypes.c_float * n_element))
    array = np.ndarray(shape=data.shape, buffer=pointer[0], dtype=np.float32)
    # Fill buffer in numpy style
    for i in range(n_element):
        array.reshape(-1)[i] = i
    # Copy data from pinned memory to another numpy array
    another_array = np.zeros(data.shape, dtype=np.float32)
    cudart.cudaMemcpy(another_array.ctypes.data, pBuffer, data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print(another_array)
    return

@case_mark
def case_use_pinned_memory():
    tw = TRTWrapperV1()

    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)
    tw.build(output_tensor_list)

    # Unpack tw.setup() and rewrite it here
    tw.runtime = trt.Runtime(tw.logger)
    tw.engine = tw.runtime.deserialize_cuda_engine(tw.engine_bytes)

    tw.tensor_name_list = [tw.engine.get_tensor_name(i) for i in range(tw.engine.num_io_tensors)]
    tw.n_input = sum([tw.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in tw.tensor_name_list])
    tw.n_output = tw.engine.num_io_tensors - tw.n_input

    tw.context = tw.engine.create_execution_context()
    for name, data in input_data.items():
        tw.context.set_input_shape(name, data.shape)

    tw.buffer = OrderedDict()
    for name in tw.tensor_name_list:
        data_type = tw.engine.get_tensor_dtype(name)
        runtime_shape = tw.context.get_tensor_shape(name)
        n_byte = trt.volume(runtime_shape) * data_type.itemsize
        device_buffer = cudart.cudaHostAlloc(n_byte, cudart.cudaHostAllocWriteCombined)[1]
        pointer = ctypes.cast(device_buffer, ctypes.POINTER(ctypes.c_float * trt.volume(runtime_shape)))
        host_buffer = np.ndarray(shape=runtime_shape, buffer=pointer[0], dtype=trt.nptype(data_type))
        tw.buffer[name] = [host_buffer, device_buffer, n_byte]

    for name, data in input_data.items():
        for i in range(data.size):
            tw.buffer[name][0].reshape(-1)[i] = data.reshape(-1)[i]

    for name in tw.tensor_name_list:
        tw.context.set_tensor_address(name, tw.buffer[name][1])

    tw.infer()

    # Deallocate pinned memory buffer manually
    for buffer in tw.buffer.values():
        cudart.cudaFreeHost(buffer[1])
    tw.buffer = None

    return

if __name__ == "__main__":
    case_pinned_memory_api()
    case_use_pinned_memory()

    print("Finish")
