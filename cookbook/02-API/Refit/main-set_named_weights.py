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

import os

import numpy as np
import tensorrt as trt
from cuda import cudart

nB, nC, nH, nW = 1, 1, 6, 9
nCOut, nKH, nKW = 1, 3, 3
data = np.tile(np.arange(1, 1 + nKH * nKW, dtype=np.float32).reshape(nKH, nKW), (nC, nH // nKH, nW // nKW)).reshape(nC, nH, nW)
weight = np.power(10, range(4, -5, -1), dtype=np.float32).reshape(nCOut, nKH, nKW)
bias = np.zeros(nCOut, dtype=np.float32)
trtFile = "./model.plan"

def run(bRefit):
    logger = trt.Logger(trt.Logger.ERROR)
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.REFIT)

        inputT0 = network.add_input("inputT0", trt.float32, (-1, nC, nH, nW))  # Dynamic Shape mode + Refit is supported since TensorRT-8.5, or we must use Static Shape model
        profile.set_shape(inputT0.name, [1, nC, nH, nW], [2, nC, nH, nW], [4, nC, nH, nW])
        config.add_optimization_profile(profile)

        fakeWeight = np.zeros([nCOut, nC, nKW, nKW], dtype=np.float32)
        fakeBias = np.zeros([nCOut], dtype=np.float32)
        convolutionLayer = network.add_convolution_nd(inputT0, nCOut, (nKH, nKW), fakeWeight, fakeBias)
        convolutionLayer.name = "conv"
        network.set_weights_name(convolutionLayer.kernel, "conv-w")
        network.set_weights_name(convolutionLayer.bias, "conv-b")

        network.mark_output(convolutionLayer.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    if bRefit == 0:
        print("Before refit ----------------------------------------------------")
    else:
        print("Refit -----------------------------------------------------------")
        refitter = trt.Refitter(engine, logger)
        refitter.set_named_weights("conv-w", weight)
        refitter.set_named_weights("conv-b", bias)

        [missingLayer, weightRole] = refitter.get_missing()  # get name and role of the missing weights
        #missingLayerList = refitter.get_missing_weights()  # only get name of the refitable weights
        for layer, role in zip(missingLayer, weightRole):
            print("[", layer, "-", role, "]")

        if refitter.refit_cuda_engine() == False:
            print("Failed Refitting engine!")
            return

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], [nB, nC, nH, nW])
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

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

    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    np.set_printoptions(precision=3, linewidth=200, suppress=True)
    cudart.cudaDeviceSynchronize()
    run(0)
    run(1)
