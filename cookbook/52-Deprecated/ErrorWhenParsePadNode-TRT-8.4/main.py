#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from cuda import cudart
import tensorrt as trt

np.random.seed(31193)
onnxFile = "./model.onnx"
trtFile = "./model.plan"
testInputShape = [1, 3, 64, 64]
testInputData = np.random.rand(np.prod(testInputShape)).astype(np.float32).reshape(testInputShape) * 2 - 1
os.system("rm -rf ./*.onnx ./*.plan")
np.set_printoptions(precision=4, linewidth=200, suppress=True)

def printArrayInfomation(x, info="", n=5):  # 打印数组统计信息
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def check(a, b, weak=False, checkEpsilon=1e-5):  # 数组逐元素比较
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))

class Pad(nn.Module):  # 原垫板节点

    def __init__(self):
        super(Pad, self).__init__()

    def forward(self, input):
        out = F.pad(input, (0, 1, 0, 2), "reflect")
        return out

class Interpolate(nn.Module):  # 用插值节点来暂时替换垫板节点

    def __init__(self):
        super(Interpolate, self).__init__()

    def forward(self, input):
        h, w = input.shape[2:]
        out = F.interpolate(input, size=[h + 2, w + 1], mode="bilinear")
        return out

inputTensor = torch.from_numpy(testInputData).cuda()  # 获取一个原模型输出用于比较计算结果
model0 = Pad().cuda()
torchOut = model0(inputTensor).detach().cpu().numpy()

model1 = Interpolate().cuda()  # 真正转模型时用插值节点

torch.onnx.export(
    model0,  # 若尝试使用 ReflectPad 来转换，报错信息见 output.txt
    inputTensor,
    onnxFile,
    input_names=["input"],
    output_names=["output"],
    verbose=True,
    keep_initializers_as_inputs=True,
    opset_version=13,
    dynamic_axes={"input": {
        0: "batch_size",
        2: "height",
        3: "width"
    }}
)
print("Succeeded convert model into ONNX!")

# TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
#os.system("trtexec --onnx=%s --saveEngine=%s --shapes=input:1x3x64x64 --buildOnly" % (onnxFile, trtFile))  # 等效的使用 trtexec 的方法

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxFile):
    print("Failed finding ONNX file!")
    exit()
print("Succeeded finding ONNX file!")
with open(onnxFile, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")

inputTensor = network.get_input(0)
profile.set_shape(inputTensor.name, (1, 3, 64, 64), (1, 3, 80, 80), (1, 3, 120, 120))
config.add_optimization_profile(profile)
"""
# network 逐层打印，用来找出 resize 层的名字
for i in range(network.num_layers):
    layer = network.get_layer(i)
    print(i, "%s,in=%d,out=%d,%s" % (str(layer.type)[10:], layer.num_inputs, layer.num_outputs, layer.name))
    for j in range(layer.num_inputs):
        tensor = layer.get_input(j)
        if tensor == None:
            print("\tInput  %2d:" % j, "None")
        else:
            print("\tInput  %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))
    for j in range(layer.num_outputs):
        tensor = layer.get_output(j)
        if tensor == None:
            print("\tOutput %2d:" % j, "None")
        else:
            print("\tOutput %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))
"""
for i in range(network.num_layers):  # 用 Slice 层替换掉 resize 层
    layer = network.get_layer(i)
    if layer.name == "Resize_22":
        sliceLayer = network.add_slice(layer.get_input(0), (0, 0, 0, 0), (1, 1, 1, 1), (1, 1, 1, 1))
        sliceLayer.set_input(2, layer.get_input(1))  # 设置新形状
        sliceLayer.mode = trt.SliceMode.REFLECT

network.unmark_output(layer.get_output(0))  # 替换模型输出张量
network.mark_output(sliceLayer.get_output(0))

engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, "wb") as f:
    f.write(engineString)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

context = engine.create_execution_context()
context.set_binding_shape(0, testInputShape)

inputH0 = np.ascontiguousarray(testInputData.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
_, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
context.execute_v2([int(inputD0), int(outputD0)])
cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
print("Succeeded running model in TensorRT!")

#printArrayInfomation(testInputData)
printArrayInfomation(torchOut, "torch")
printArrayInfomation(outputH0, "tensorrt")
check(torchOut, outputH0, True)
