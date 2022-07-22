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
import numpy as np
from datetime import datetime as dt
import torch as t
from torch.autograd import Variable
from cuda import cudart
import tensorrt as trt

nTrainBatchSize = 128
ptFile = "./model.pt"
onnxFile = "./model.onnx"
onnxSurgeonFile = "./model-surgeon.onnx"
trtFile = "./model.plan"
nB, nC, nH, nW = 2, 3, 4, 5
inputX = np.random.rand(nB, nC, nH, nW).astype(np.float32).reshape([nB, nC, nH, nW])

os.system("rm -rf ./*.pt ./*.onnx ./*.plan")
np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

# pyTorch 中创建网络--------------------------------------------------------------
class Net(t.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        x = t.mul(x, 1)
        x = t.nonzero(x)
        y = t.mul(x, 1)
        return y

net = Net().cuda()
t.save(net, ptFile)
print("Succeeded building model in pyTorch!")

# 将 .pt 文件转换为 .onnx 文件 ----------------------------------------------------
t.onnx.export(net, t.randn(nB, nC, nH, nW, device="cuda"), onnxFile, example_outputs=[t.randn(nB, nC, 1, 1, device="cuda")], input_names=['x'], output_names=['y'], do_constant_folding=True, verbose=True, keep_initializers_as_inputs=True, opset_version=12, dynamic_axes={"x": {0: "nBatchSize"}})
print("Succeeded converting model into onnx!")

# TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
if os.path.isfile(trtFile):
    with open(trtFile, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine == None:
        print("Failed loading engine!")
        exit()
    print("Succeeded loading engine!")
else:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, [1, nC, nH, nW], [nB, nC, nH, nW], [nB * 2, nC, nH, nW])
    config.add_optimization_profile(profile)
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, "wb") as f:
        f.write(engineString)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

context = engine.create_execution_context()
context.set_binding_shape(0, [nB, nC, nH, nW])
_, stream = cudart.cudaStreamCreate()
print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))

data = np.random.rand(nB, nC, nH, nW).astype(np.float32)
inputH0 = np.ascontiguousarray(data.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
_, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
cudart.cudaStreamSynchronize(stream)

print("inputH0 :", data.shape)
#print(data)
print("outputH0:", outputH0.shape)
print(outputH0)

cudart.cudaStreamDestroy(stream)
cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
print("Succeeded running model in TensorRT!")
