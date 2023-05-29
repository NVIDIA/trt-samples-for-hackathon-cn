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
import tensorrt as trt

np.random.seed(31193)
m, k, n = 3, 4, 5
data0 = np.tile(np.arange(1, 1 + k), [m, 1]) * 1 / 10 ** (2 * np.arange(1, 1 + m) - 2)[:, np.newaxis]
data1 = np.tile(np.arange(k), [n, 1]).T * 10 ** np.arange(n)[np.newaxis, :]

def run(useFP16):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    if useFP16:
        config.flags = config.flags | (1 << int(trt.BuilderFlag.STRICT_TYPES)) | (1 << int(trt.BuilderFlag.FP16))

    inputT0 = network.add_input("inputT0", trt.float32, (m, k))

    constantLayer = network.add_constant([k, n], np.ascontiguousarray(data1.astype(np.float16 if useFP16 else np.float32)))
    matrixMultiplyLayer = network.add_matrix_multiply(inputT0, trt.MatrixOperation.NONE, constantLayer.get_output(0), trt.MatrixOperation.NONE)
    if useFP16:
        matrixMultiplyLayer.precision = trt.float16
        matrixMultiplyLayer.get_output(0).dtype = trt.float16

    network.mark_output(matrixMultiplyLayer.get_output(0))
    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    bufferH.append(np.ascontiguousarray(data0))
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
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    cudart.cudaDeviceSynchronize()

    run(False)  # using FP32
    run(True)  # using FP16
