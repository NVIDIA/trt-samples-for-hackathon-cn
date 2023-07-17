#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
#

from time import time_ns

import numpy as np
import tensorrt as trt
from cuda import cudart

np.random.seed(31193)

shape = [4, 1024, 64]
data = np.random.rand(*shape).reshape(shape).astype(np.float32) * 2 - 1

def run(nLevel):
    testCase = "<Level=%d>" % (nLevel)
    trtFile = "model-Level%d.plan" % (nLevel)
    print("Test %s" % testCase)

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.builder_optimization_level = nLevel

    inputTensor = network.add_input("inputT0", trt.float32, [-1] + shape[1:])  # I write a "complex" network to see the performance differences
    profile.set_shape(inputTensor.name, [1] + shape[1:], shape, [16] + shape[1:])
    config.add_optimization_profile(profile)

    _0 = inputTensor
    for i in range(64, 256):
        w = np.random.rand(1, i, i + 1).astype(np.float32)
        b = np.random.rand(1, 1, i + 1).astype(np.float32)
        _1 = network.add_constant(w.shape, trt.Weights(np.ascontiguousarray(w)))
        _2 = network.add_matrix_multiply(_0, trt.MatrixOperation.NONE, _1.get_output(0), trt.MatrixOperation.NONE)
        _3 = network.add_constant(b.shape, trt.Weights(np.ascontiguousarray(b)))
        _4 = network.add_elementwise(_2.get_output(0), _3.get_output(0), trt.ElementWiseOperation.SUM)
        _5 = network.add_activation(_4.get_output(0), trt.ActivationType.RELU)
        _0 = _5.get_output(0)

    network.mark_output(_0)

    t0 = time_ns()
    engineString = builder.build_serialized_network(network, config)
    t1 = time_ns()
    print("Time of building: %fms" % ((t1 - t0) / (10 ** 6)))

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], shape)

    bufferH = []
    bufferH.append(np.ascontiguousarray(data))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    # warming up
    context.execute_async_v3(0)
    cudart.cudaDeviceSynchronize()

    t0 = time_ns()
    for _ in range(10):
        context.execute_async_v3(0)
    cudart.cudaDeviceSynchronize()
    t1 = time_ns()
    print("Time of inference: %fms" % ((t1 - t0) / (10 ** 6)))

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for b in bufferD:
        cudart.cudaFree(b)

    print("Test %s finish!\n" % testCase)

if __name__ == "__main__":
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)
    run(5)
