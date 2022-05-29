#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import sys
import cv2
import numpy as np
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt

dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
sys.path.append(dataPath)

np.random.seed(97)
nTrainbatchSize = 128
pbFile = "./model-NCHW.pb"
caffePrototxtFile = "./model.prototxt"
caffeModelFile = "./model.caffemodel"
trtFile = "./model-NCHW.plan"
inputImage = dataPath + '8.png'

np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

# TensorRT 中加载 Caffe 模型并创建 engine -----------------------------------------
logger = trt.Logger(trt.Logger.VERBOSE)
if os.path.isfile(trtFile):
    with open(trtFile, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine == None:
        print("Failed loading engine!")
        exit()
    print("Succeeded loading engine!")
else:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = 3 << 30
    parser = trt.CaffeParser()
    with open(caffePrototxtFile, 'rb') as f0, open(caffeModelFile, 'rb') as f1:
        net = parser.parse_buffer(f0.read(), f1.read(), network, trt.float32)
        if net is None:
            print("Failed parsing caffe file!")
        print("Succeeded parsing cafe file!")

    outTensor = net.find('y')  # 找到网络的输出层
    squeezeLayer = network.add_reduce(outTensor, trt.ReduceOperation.SUM, (1 << 2) + (1 << 3), False)  # 删掉先前手工添加的、多余的维度
    argmaxLayer = network.add_topk(squeezeLayer.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)  # 补上 Caffe 不支持的 Argmax 层

    network.mark_output(argmaxLayer.get_output(1))
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write(engineString)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

context = engine.create_execution_context()
print("Binding0->", engine.get_binding_shape(0), context.get_binding_shape(0), engine.get_binding_dtype(0))
print("Binding1->", engine.get_binding_shape(1), context.get_binding_shape(1), engine.get_binding_dtype(1))

data = cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE).astype(np.float32)
inputH0 = np.ascontiguousarray(data.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
inputD0 = cudart.cudaMalloc(inputH0.nbytes)[1]
outputD0 = cudart.cudaMalloc(outputH0.nbytes)[1]

cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
context.execute_v2([int(inputD0), int(outputD0)])
cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

print("inputH0 :", data.shape)
#print(data)
print("outputH0:", outputH0.shape)
print(outputH0)

cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
print("Succeeded running model in TensorRT!")
