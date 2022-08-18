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
import ctypes
import numpy as np
import torch as t
import onnx
import onnx_graphsurgeon as gs
from cuda import cudart
import tensorrt as trt

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
print("Succeeded converting model into onnx!")

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
os.system("make")
print("Succeeded building LayerNorm Plugin!")

# TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
ctypes.cdll.LoadLibrary(soFile)
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
    if not os.path.exists(onnxSurgeonFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
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
print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))

inputH0 = np.ascontiguousarray(inputX.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
_, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
context.execute_v2([int(inputD0), int(outputD0)])
cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

#printArrayInfomation(outputPyTorch)
#printArrayInfomation(outputH0)
check(outputH0, outputPyTorch, True)

cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
print("Succeeded running model in TensorRT!")
