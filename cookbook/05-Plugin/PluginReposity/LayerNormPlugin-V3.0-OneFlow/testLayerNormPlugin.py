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
import ctypes
import numpy as np
import torch as t
import onnx
import onnx_graphsurgeon as gs
from cuda import cudart
import tensorrt as trt
import time

ptFile = "./model.pt"
onnxFile = "./model.onnx"
onnxSurgeonFile = "./model-surgeon.onnx"
soFile = "./LayerNormPlugin.so"
trtFile = "./model.plan"
nIn, tIn, cIn = 1024,256, 256
epsilon = 1e-5

np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

# pyTorch 中创建网络并保存为 .pt 文件 ----------------------------------------------
class Net(t.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.LayerNorm = t.nn.LayerNorm(cIn, elementwise_affine=False, eps=epsilon)

    def forward(self, x):
        x = t.mul(x, 1)
        x = self.LayerNorm(x)
        y = t.mul(x, 1)
        return y

net = Net().cuda()
t.save(net, ptFile)
print("Succeeded building model in pyTorch!")

# 将 .pt 文件转换为 .onnx 文件 ----------------------------------------------------
t.onnx.export(
    net,
    t.randn(nIn, tIn, cIn, device="cuda"),
    onnxFile,
    # example_outputs=[t.randn(nIn, cIn, 1, 1, device="cuda")],
    input_names=['x'],
    output_names=['y'],
    #do_constant_folding=True,
    verbose=False,
    keep_initializers_as_inputs=True,
    opset_version=12,
    dynamic_axes={"x": {
        0: "nBatchSize"
    }}
)
print("Succeeded converting model into onnx!")

# 在 .onnx 文件中将 LayerNorm 模块替换为 Plugin ------------------------------------
graph = gs.import_onnx(onnx.load(onnxFile))
graph.inputs[0].shape = ['bs', tIn, cIn]
graph.outputs[0].shape = ['bs', tIn, cIn]

nLayerNorm = 0
for node in graph.nodes:
    if node.op == 'Div':
        nLayerNorm += 1
        # pluginVariable = gs.Variable("MyLayerNorm-%d" % nLayerNorm, np.dtype(np.float32), None)
        pluginNode = gs.Node("LayerNorm", "MyLayerNorm-%d" % nLayerNorm, inputs=[node.i(0).inputs[0]], outputs=[node.outputs[0]], attrs={"epsilon": 1e-5})
        graph.nodes.append(pluginNode)
        # node.outputs[0].inputs[0] = pluginVariable
        node.outputs.clear()

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnxSurgeonFile)
print("Succeeded replacing LayerNorm Plugin node!")

# 编译 Plugin 为 .so 文件 --------------------------------------------------------
# os.system("make")
# print("Succeeded building LayerNorm Plugin!")

# TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
ctypes.cdll.LoadLibrary(soFile)

builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.max_workspace_size = 3 << 30

config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.STRICT_TYPES)

parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxSurgeonFile):
    print("Failed finding onnx file!")
    exit()
print("Succeeded finding onnx file!")
with open(onnxSurgeonFile, 'rb') as model:
    if not parser.parse(model.read()):
        print("Failed parsing onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing onnx file!")


# for i, layer in enumerate(network):
#     if "MyLayerNorm" in layer.name:
#         layer.precision = trt.DataType.FLOAT
#         print("find layernorm")

inputTensor = network.get_input(0)
profile.set_shape(inputTensor.name, [1, tIn, cIn], [nIn, tIn, cIn], [nIn, tIn, cIn])
config.add_optimization_profile(profile)
engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, 'wb') as f:
    f.write(engineString)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

context = engine.create_execution_context()
context.set_binding_shape(0, [nIn, tIn, cIn])
_, stream = cudart.cudaStreamCreate()
print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))


seed = np.random.randn(nIn * cIn * tIn)
data = (2*np.sin(seed) + 3*np.exp(seed)).reshape(nIn, tIn, cIn).astype(np.float32)
# data = np.arange(nIn * cIn * tIn).reshape(nIn, tIn, cIn).astype(np.float32) ** 2
# data = np.zeros((nIn, tIn, cIn)).astype(np.float32)

start_time = time.time()
inputH0 = np.ascontiguousarray(data.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
_, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
cudart.cudaStreamSynchronize(stream)

print("inputH0 :", data.shape)
print("outputH0:", outputH0.shape)

from time import time_ns

for i in range(10):
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cudart.cudaStreamSynchronize(stream)
        
time0 = time_ns()
for i in range(100):
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cudart.cudaStreamSynchronize(stream)
time1 = time_ns()
print("average %fms per inference\n"%((time1-time0)/100/1000000))



cudart.cudaStreamDestroy(stream)


net.eval()
torch_out = net(t.tensor(data)).detach().cpu().numpy()
outputH0 = outputH0.reshape(-1)
torch_out = torch_out.reshape(-1)

diff_map = ~np.isclose(outputH0, torch_out, atol=1e-3)
ind = diff_map.nonzero()[0]

print(len(ind))
print(ind[:50])
print(outputH0[ind[:50]])
print(torch_out[ind[:50]])

print(np.linalg.norm(outputH0 - torch_out))
