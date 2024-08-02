#
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
#

import sys
from collections import OrderedDict

import numpy as np
import tensorrt as trt
from cuda import cudart

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperShapeInput

shape = [3, 4, 5]
input_data = {}
input_data["inputT0"] = np.zeros(np.prod(shape), dtype=np.float32).reshape(shape)  # Execution input tensor
input_data["inputT1"] = np.array(shape, dtype=np.int32)  # Shape input tensor

tw = TRTWrapperShapeInput()

tensor0 = tw.network.add_input("inputT0", trt.float32, [-1 for _ in shape])
tensor1 = tw.network.add_input("inputT1", trt.int32, [len(shape)])
tw.profile.set_shape(tensor0.name, [1 for _ in shape], shape, shape)
tw.profile.set_shape_input(tensor1.name, [1 for _ in shape], shape, shape)
tw.config.add_optimization_profile(tw.profile)

layer0 = tw.network.add_identity(tensor0)
tensor0 = layer0.get_output(0)
tensor0.name = "outputT0"
layer1 = tw.network.add_identity(tensor1)
tensor1 = layer1.get_output(0)
tensor1.name = "outputT1"
tw.build([tensor0, tensor1])

# We work similar as TRTWrapperShapeInput's setup() and infer()
tw.runtime = trt.Runtime(tw.logger)
tw.engine = tw.runtime.deserialize_cuda_engine(tw.engine_bytes)
tw.tensor_name_list = [tw.engine.get_tensor_name(i) for i in range(tw.engine.num_io_tensors)]

context = tw.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
# Alternative values of trt.ExecutionContextAllocationStrategy:
# trt.ExecutionContextAllocationStrategy.STATIC             -> 0, default value
# trt.ExecutionContextAllocationStrategy.ON_PROFILE_CHANGE  -> 1
# trt.ExecutionContextAllocationStrategy.USER_MANAGED       -> 2

#context.device_memory = 0  # We need to set GPU memory pointer to context when using trt.ExecutionContextAllocationStrategy.USER_MANAGED

print(f"{context.name  = }")
print(f"{context.engine = }")
print(f"{context.error_recorder = }")  # -> 04-Feature/ErrorRecorder
context.set_aux_streams([])
context.set_optimization_profile_async(0, 0)
print(f"{context.active_optimization_profile  = }")

print(f"{context.enqueue_emits_profile = }")
print(f"{context.persistent_cache_limit  = }")
print(f"{context.get_input_consumed_event()  = }")
print(f"{context.nvtx_verbosity  = }")
# Alternative values of trt.ProfilingVerbosity:
# trt.ProfilingVerbosity.LAYER_NAMES_ONLY   -> 0, default
# trt.ProfilingVerbosity.NONE               -> 1
# trt.ProfilingVerbosity.DETAILED           -> 2

print("Before setting shape")
print(f"{context.infer_shapes() = }")

for name in tw.tensor_name_list:
    mode = tw.engine.get_tensor_mode(name)
    data_type = tw.engine.get_tensor_dtype(name)
    runtime_shape = context.get_tensor_shape(name)
    #runtime_strides = context.get_tensor_strides(name)  # invalid here
    runtime_address = context.get_tensor_address(name)
    if mode == trt.TensorIOMode.OUTPUT:
        debug_state = context.get_debug_state(name)
        #max_output_size = context.get_max_output_size(name)  # invalid here
    else:
        debug_state = None
        max_output_size = None
    print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}-> {data_type}, {runtime_shape}, {runtime_address}, {debug_state}, {name}")

for name, data in input_data.items():
    if tw.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
        context.set_input_shape(name, data.shape)
    else:
        context.set_tensor_address(name, data.ctypes.data)

tw.buffer = OrderedDict()
for name in tw.tensor_name_list:
    data_type = tw.engine.get_tensor_dtype(name)
    runtime_shape = context.get_tensor_shape(name)
    n_byte = trt.volume(runtime_shape) * data_type.itemsize
    host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
    if tw.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
        device_buffer = cudart.cudaMalloc(n_byte)[1]
    else:
        device_buffer = None
    tw.buffer[name] = [host_buffer, device_buffer, n_byte]

for name, data in input_data.items():
    tw.buffer[name][0] = np.ascontiguousarray(data)

for name in tw.tensor_name_list:
    if tw.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
        context.set_tensor_address(name, tw.buffer[name][1])
    elif tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
        context.set_tensor_address(name, tw.buffer[name][0].ctypes.data)

print("After setting shape")
print(f"{context.infer_shapes() = }")

for name in tw.tensor_name_list:
    mode = tw.engine.get_tensor_mode(name)
    data_type = tw.engine.get_tensor_dtype(name)
    runtime_shape = context.get_tensor_shape(name)
    runtime_strides = context.get_tensor_strides(name)
    runtime_address = context.get_tensor_address(name)
    if mode == trt.TensorIOMode.OUTPUT:
        debug_state = context.get_debug_state(name)
        max_output_size = context.get_max_output_size(name)
    else:
        debug_state = None
        max_output_size = None
    print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}-> {data_type}, {runtime_shape}, {runtime_strides}, {runtime_address}, {debug_state}, {max_output_size}, {name}")

print(f"{context.all_binding_shapes_specified = }")
print(f"{context.all_shape_inputs_specified = }")
print(f"{context.debug_sync = }")
print(f"{context.update_device_memory_size_for_shapes() = }")  # work when creating context with trt.ExecutionContextAllocationStrategy.USER_MANAGED

for name in tw.tensor_name_list:
    if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT and tw.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
        cudart.cudaMemcpy(tw.buffer[name][1], tw.buffer[name][0].ctypes.data, tw.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

context.execute_async_v3(0)  # asynchonzied execution
context.execute_v2([tw.buffer[name][1] for name in tw.tensor_name_list])  # synchonzied execution

for _, device_buffer, _ in tw.buffer.values():
    cudart.cudaFree(device_buffer)

print("Finish")
"""
API not showed:
get_debug_listener          -> 04-Feature/DebugTensor
get_debug_state             -> 04-Feature/DebugTensor
set_all_tensors_debug_state -> 04-Feature/DebugTensor
set_debug_listener          -> 04-Feature/DebugTensor
set_tensor_debug_state      -> 04-Feature/DebugTensor

profiler                    -> 04-Feature/Profiler
report_to_profiler          -> 04-Feature/Profiler

get_input_consumed_event    -> 04-Feature/Event
set_input_consumed_event    -> 04-Feature/Event

temporary_allocator         -> 04-Feature/GpuAllocator

get_output_allocator        -> 04-Feature/OutputAllocator
set_output_allocator        -> 04-Feature/OutputAllocator
"""
