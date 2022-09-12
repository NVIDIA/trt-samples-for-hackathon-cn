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

import ctypes
from cuda import cudart
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import os
import tensorrt as trt
import torch as t

onnxFile = "./model.onnx"
onnxSurgeonFile = "./model-surgeon.onnx"
soFile = "./LayerNormPlugin.so"
trtFile = "./model.plan"
nBS = 16
nSL = 64
nEmbedding = 256
epsilon = 1e-5
inputX = np.random.rand(nBS, nSL, nEmbedding).astype(np.float32).reshape(nBS, nSL, nEmbedding)

np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

def printArrayInfomation(x, info="", n=5):
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:", res, diff0, diff1)

# pyTorch 中导出网络为 .onnx 文件 -------------------------------------------------
class Net(t.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.LayerNorm = t.nn.LayerNorm(nEmbedding, elementwise_affine=False, eps=epsilon)

    def forward(self, x):
        x = t.mul(x, 1.0)
        x = self.LayerNorm(x)
        y = t.mul(x, 1.0)
        return y

net = Net().cuda()
outputPyTorch = net(t.from_numpy(inputX).cuda()).detach().cpu().numpy()
t.onnx.export(
    net,
    t.from_numpy(inputX).cuda(),
    onnxFile,
    input_names=["x"],
    output_names=["y"],
    #do_constant_folding=True,
    verbose=True,
    keep_initializers_as_inputs=True,
    opset_version=13,
    dynamic_axes={"x": {
        0: "nBS",
        1: "nSL"
    }}
)
print("Succeeded converting model into ONNX!")

# 在 .onnx 文件中将 LayerNorm 模块替换为 Plugin ------------------------------------
graph = gs.import_onnx(onnx.load(onnxFile))
graph.inputs[0].shape = ["nBS", "nSL", nEmbedding]
graph.outputs[0].shape = ["nBS", "nSL", nEmbedding]

nLayerNorm = 0
for node in graph.nodes:
    if node.op == "Div":
        nLayerNorm += 1
        pluginVariable = gs.Variable("MyLayerNorm-%d" % nLayerNorm, np.dtype(np.float32), None)
        pluginNode = gs.Node("LayerNorm", "MyLayerNorm-%d" % nLayerNorm, inputs=[node.i(0).i(0).outputs[0]], outputs=[pluginVariable], attrs={"epsilon": node.i(1).i().i(1).attrs["value"].values.reshape(1)})
        graph.nodes.append(pluginNode)
        node.o().inputs[0] = pluginVariable
        node.outputs.clear()

graph.cleanup()
onnx.save(gs.export_onnx(graph), onnxSurgeonFile)
print("Succeeded replacing LayerNorm Plugin node!")

# 编译 Plugin 为 .so 文件 --------------------------------------------------------
#os.system("make")
#print("Succeeded building LayerNorm Plugin!")

# TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
ctypes.cdll.LoadLibrary(soFile)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxSurgeonFile):
    print("Failed finding ONNX file!")
    exit()
print("Succeeded finding ONNX file!")
with open(onnxSurgeonFile, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")

inputTensor = network.get_input(0)
profile.set_shape(inputTensor.name, [1, 1, nEmbedding], [nBS, nSL, nEmbedding], [nBS * 2, nSL * 2, nEmbedding])
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
context.set_binding_shape(0, [nBS, nSL, nEmbedding])
#print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
#for i in range(nInput):
#    print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
#for i in range(nInput, nInput + nOutput):
#    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

bufferH = []
bufferH.append(np.ascontiguousarray(inputX))
for i in range(nOutput):
    bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
bufferD = []
for i in range(engine.num_bindings):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

context.execute_v2(bufferD)

for i in range(nOutput):
    cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

#printArrayInfomation(outputPyTorch)
#printArrayInfomation(outputH0)
check(bufferH[-1], outputPyTorch, True)

for buffer in bufferD:
    cudart.cudaFree(buffer)

print("Succeeded running model in TensorRT!")