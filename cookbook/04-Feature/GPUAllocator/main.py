# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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

from tensorrt_cookbook import (MyGpuAllocator, TRTWrapperV1, build_mnist_network_trt)

shape = [1, 1, 28, 28]
data = {"x": np.random.rand(np.prod(shape)).astype(np.float32).reshape(shape) * 2 - 1}

tw = TRTWrapperV1()

output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)
tw.build(output_tensor_list)

# Work similar as TRTWrapperV1.setup()
myGpuAllocator = MyGpuAllocator(log=True)

tw.runtime = trt.Runtime(tw.logger)
tw.runtime.gpu_allocator = myGpuAllocator  # can be assign GPU Allocator to Runtime or ExecutionContext

# Work similar as TRTWrapperV1.setup()
tw.engine = tw.runtime.deserialize_cuda_engine(tw.engine_bytes)

tw.tensor_name_list = [tw.engine.get_tensor_name(i) for i in range(tw.engine.num_io_tensors)]
tw.n_input = sum([tw.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in tw.tensor_name_list])
tw.n_output = tw.engine.num_io_tensors - tw.n_input

tw.context = tw.engine.create_execution_context()
tw.context.temporary_allocator = myGpuAllocator  # can be assign GPU Allocator to Runtime or ExecutionContext

for name, value in data.items():
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

for name, value in data.items():
    tw.buffer[name][0] = np.ascontiguousarray(value)

for name in tw.tensor_name_list:
    tw.context.set_tensor_address(name, tw.buffer[name][1])

tw.infer(b_print_io=False)

print("After infernece")
