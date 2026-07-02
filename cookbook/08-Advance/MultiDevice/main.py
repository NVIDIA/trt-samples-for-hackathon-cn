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

from collections import OrderedDict  # keep the order of the tensors implicitly

import cuda.bindings.runtime as cudart
import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, datatype_cast

REQUIRED_WORLD_SIZE = 2

data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5)}

if __name__ == "__main__":

    _, device_count = cudart.cudaGetDeviceCount()
    if device_count < REQUIRED_WORLD_SIZE:
        print(f"Skip since no enough GPU is ready (need {REQUIRED_WORLD_SIZE}, get {device_count})")
        exit(0)

    # Always use device 0 to build engine
    cudart.cudaSetDevice(0)
    print(f"Dvice = {cudart.cudaGetDevice()[1]}")

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    weight_shape = data["tensor"].transpose(0, 1, 3, 2).shape
    layer_weight = tw.network.add_constant(weight_shape, trt.Weights(np.ascontiguousarray(np.ones(weight_shape, dtype=np.float32))))
    layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.NONE)

    tw.build([layer.get_output(0)])

    # Use device 0 to do normal inference once
    tw.setup(data)
    tw.infer()

    # Try to run on device 1
    cudart.cudaSetDevice(1)
    print(f"Dvice = {cudart.cudaGetDevice()[1]}")

    print(f"Use tw.engine_bytes")  # OK
    tw_device1 = TRTWrapperV1()
    tw_device1.runtime = trt.Runtime(tw.logger)
    tw_device1.engine = tw_device1.runtime.deserialize_cuda_engine(tw.engine_bytes)
    tw_device1.context = tw_device1.engine.create_execution_context()
    tw_device1.setup(data)
    tw_device1.infer()

    # print(f"Use tw.engine")
    # Get error information like below, so we know the engine cannot be shared across devices
    # [TRT] [E] [graphContext.cpp::~MyelinGraphContext::101] Error Code 1: Myelin ([impl.cpp:675: unload_cuda] Error 700 destroying event '0x4e0828c0'. In ~MyelinGraphContext at /_src/runtime/myelin/graphContext.cpp:101)
    # tw_device1 = TRTWrapperV1()
    # tw_device1.runtime = trt.Runtime(tw.logger)
    # tw_device1.engine = tw.engine
    # tw_device1.context = tw_device1.engine.create_execution_context()
    # tw_device1.setup(data)
    # tw_device1.infer()

    print("Finish")
