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

import os
from collections import OrderedDict
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from pathlib import Path

trt_file = Path("model.trt")
input_tensor_name = "inputT0"
data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    if os.path.isfile(trt_file):
        with open(trt_file, "rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Fail getting serialized engine")
            return
        print("Succeed getting serialized engine")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        builder_config = builder.create_builder_config()
        builder_config.set_flag(trt.BuilderFlag.SAFETY_SCOPE)  # use Safety mode
        builder_config.engine_capability = trt.EngineCapability.SAFETY  # Error when adding this:
        # [TRT] [E] IRuntime::deserializeCudaEngine: Error Code 1: Serialization (Serialization assertion header.magicTag == kEXPECTED_MAGIC_TAG failed.Trying to load an engine created with incompatible serialization version (1297697870 != 1953657958). Check that the engine was not created using safety runtime, same OS was used and version compatibility parameters were set accordingly and that it is a TRT engine file. In throwUnlessHeaderOk at /_src/runtime/dispatch/runtime.cpp:42)

        inputTensor = network.add_input("inputT0", trt.float32, [3, 4, 5])  # only Explicit Batch + Static Shape is supported in safety mode

        identityLayer = network.add_identity(inputTensor)
        network.mark_output(identityLayer.get_output(0))

        engineString = builder.build_serialized_network(network, builder_config)
        if engineString == None:
            print("Fail building serialized engine")
            return
        print("Succeed building serialized engine")
        with open(trt_file, "wb") as f:
            f.write(engineString)
            print("Succeed saving .trt file")

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    print("Engine Capability:", engine.engine_capability)
    if engine == None:
        print("Fail building engine")
        return
    print("Succeed building engine")

    context = engine.create_execution_context()
    tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    context.set_input_shape(input_tensor_name, data.shape)

    for name in tensor_name_list:
        mode = engine.get_tensor_mode(name)
        data_type = engine.get_tensor_dtype(name)
        buildtime_shape = engine.get_tensor_shape(name)
        runtime_shape = context.get_tensor_shape(name)
        print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {runtime_shape}, {name}")

    buffer = OrderedDict()
    for name in tensor_name_list:
        data_type = engine.get_tensor_dtype(name)
        runtime_shape = context.get_tensor_shape(name)
        n_byte = trt.volume(runtime_shape) * np.dtype(trt.nptype(data_type)).itemsize
        host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
        device_buffer = cudart.cudaMalloc(n_byte)[1]
        buffer[name] = [host_buffer, device_buffer, n_byte]

    buffer[input_tensor_name][0] = np.ascontiguousarray(data)

    for name in tensor_name_list:
        context.set_tensor_address(name, buffer[name][1])

    for name in tensor_name_list:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cudart.cudaMemcpy(buffer[name][1], buffer[name][0].ctypes.data, buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_async_v3(0)

    for name in tensor_name_list:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            cudart.cudaMemcpy(buffer[name][0].ctypes.data, buffer[name][1], buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for name in tensor_name_list:
        print(name)
        print(buffer[name][0])

    for _, device_buffer, _ in buffer.values():
        cudart.cudaFree(device_buffer)

if __name__ == "__main__":
    trt_file.unlink(missing_ok=True)

    # run()  # TODO: fix this
    # run()

    print("Finish")
