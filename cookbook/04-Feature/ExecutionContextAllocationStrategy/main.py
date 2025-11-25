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

import tensorrt as trt
from cuda.bindings import runtime as cudart

from tensorrt_cookbook import TRTWrapperV1, build_mnist_network_trt, case_mark

@case_mark
def case_allocation_strategy_static(tw):
    # Always use maximum context memory for all profile (default strategy)
    tw.context = tw.engine.create_execution_context()

@case_mark
def case_allocation_strategy_on_profile_change(tw):
    # Reallocate GPU memory when changing profile
    tw.context = tw.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.ON_PROFILE_CHANGE)

    tw.context.set_optimization_profile_async(0, 0)

    tw.context.set_optimization_profile_async(1, 0)

@case_mark
def case_allocation_strategy_user_managed(tw):
    # Use GPU memory which user provide
    tw.context = tw.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)

    tw.context.set_optimization_profile_async(0, 0)
    address = cudart.cudaMalloc(tw.engine.get_device_memory_size_for_profile_v2(0))[1]
    #address = cudart.cudaMalloc(tw.engine.get_device_memory_size_for_profile(0))[1]
    tw.context.device_memory = address

    tw.context.set_optimization_profile_async(1, 0)
    address1 = cudart.cudaMalloc(tw.engine.get_device_memory_size_for_profile_v2(1))[1]
    #address1 = cudart.cudaMalloc(tw.engine.get_device_memory_size_for_profile(1))[1]
    tw.context.device_memory = address1

    cudart.cudaFree(address)
    cudart.cudaFree(address1)

if __name__ == "__main__":

    tw = TRTWrapperV1()

    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)

    # Add another profile
    tensor = tw.network.get_input(0)
    profile1 = tw.builder.create_optimization_profile()
    profile1.set_shape(tensor.name, (256, ) + tensor.shape[1:], (256, ) + tensor.shape[1:], (512, ) + tensor.shape[1:])
    tw.config.add_optimization_profile(tw.profile)

    tw.build(output_tensor_list)

    tw.engine = trt.Runtime(tw.logger).deserialize_cuda_engine(tw.engine_bytes)

    tw.tensor_name_list = [tw.engine.get_tensor_name(i) for i in range(tw.engine.num_io_tensors)]
    print(f"{tw.engine.device_memory_size_v2 = }B")
    print(f"{tw.engine.get_device_memory_size_for_profile_v2(0) = }B")
    print(f"{tw.engine.get_device_memory_size_for_profile_v2(1) = }B")

    case_allocation_strategy_static(tw)
    case_allocation_strategy_on_profile_change(tw)
    case_allocation_strategy_user_managed(tw)

    print("Finish")
