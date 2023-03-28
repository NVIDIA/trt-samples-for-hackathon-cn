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

from cuda import cudart
import numpy as np
import os
import tensorrt as trt
from time import time_ns

trtFile = "./model.plan"
nB, nM, nN = 4, 32, 1024
nLoop = 10
nWarmUp = 10
nTest = 100
np.random.seed(31193)

weightUp = (np.random.rand(nM, nN).astype(np.float32) * 2 - 1)
weightDown = (np.random.rand(nN, nM).astype(np.float32) * 2 - 1)

weightUp = weightUp.reshape(-1)
for i in range(0, weightUp.shape[0], 2):
    weightUp[i] = 0
weightUp = weightUp.reshape(nM, nN)

print(weightUp)

weightDown = weightDown.reshape(-1)
for i in range(0, weightDown.shape[0], 2):
    weightDown[i] = 0
weightDown = weightDown.reshape(nN, nM)

print(weightDown)

def printArrayInfomation(x, info="", n=5):
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def run(bUseSparsity):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    if bUseSparsity:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    inputTensor = network.add_input("inputT0", trt.float32, [-1, nM])
    profile.set_shape(inputTensor.name, [1, nM], [nB, nM], [nB * 2, nM])
    config.add_optimization_profile(profile)

    constantLayer0 = network.add_constant(weightUp.shape, trt.Weights(np.ascontiguousarray(weightUp)))
    constantLayer1 = network.add_constant(weightDown.shape, trt.Weights(np.ascontiguousarray(weightDown)))

    tensor = inputTensor
    for i in range(nLoop):
        layer0 = network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, constantLayer0.get_output(0), trt.MatrixOperation.NONE)
        layer1 = network.add_activation(layer0.get_output(0), trt.ActivationType.RELU)
        tensor = layer1.get_output(0)

        layer2 = network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, constantLayer1.get_output(0), trt.MatrixOperation.NONE)
        layer3 = network.add_activation(layer2.get_output(0), trt.ActivationType.RELU)
        tensor = layer3.get_output(0)

    network.mark_output(tensor)
    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], [nB, nM])
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    data = np.arange(np.prod([nB, nM]), dtype=np.float32).reshape(nB, nM) * 2 - 1
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

    context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nIO):
        print(lTensorName[i])
        print(bufferH[i])

    for i in range(nWarmUp):
        context.execute_async_v3(0)

    t0 = time_ns()
    for i in range(nTest):
        context.execute_async_v3(0)
    t1 = time_ns()
    print("Time per inference: %f ms" % ((t1 - t0) / 1000000 / nTest))

    printArrayInfomation(bufferH[-1])

    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    run(False)
    run(True)
