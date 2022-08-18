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
from datetime import datetime as dt
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import os

os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf1
import tensorrt as trt

tf1.compat.v1.disable_eager_execution()
np.random.seed(97)
tf1.compat.v1.set_random_seed(97)
epsilon = 1e-6
pbFile = "./model.pb"
onnxFile = "./model.onnx"
onnxSurgeonFile = "./model-surgeon.onnx"
soFile = "./AddScalarPlugin.so"
trtFile = "./model.plan"
nB, nC, nH, nW = 2, 3, 4, 5
inputX = np.random.rand(nB, nC, nH, nW).astype(np.float32).reshape([nB, nC, nH, nW])

os.system("rm -rf ./*.pb ./*.onnx ./*.plan ./*.o ./*.d ./*.so")
np.set_printoptions(precision=4, linewidth=200, suppress=True)
tf1.compat.v1.disable_eager_execution()
cudart.cudaDeviceSynchronize()

def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:", res, diff0, diff1)

def printArrayInfomation(x, info="", n=5):  # 用于输出数组统计信息
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

# TensorFlow 中创建网络并保存为 .pb 文件 -------------------------------------------
x = tf1.compat.v1.placeholder(tf1.float32, [None, nC, nH, nW], name="x")
_h1 = tf1.multiply(x, 1, name="node-0")  # 某些前处理
_h2 = tf1.add(_h1, 1, name="node-1")  # 想要替换的算子 / 模块
y = tf1.multiply(_h2, 1, name="node-2")  # 某些后处理

tfConfig = tf1.compat.v1.ConfigProto()
tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf1.compat.v1.Session(config=tfConfig)
sess.run(tf1.compat.v1.global_variables_initializer())
outputTF = sess.run(y, feed_dict={x: inputX})

constantGraph = tf1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["node-2"])
with tf1.gfile.FastGFile(pbFile, mode="wb") as f:
    f.write(constantGraph.SerializeToString())

sess.close()
print("Succeeded building model in TensorFlow1!")

# 将 .pb 文件转换为 .onnx 文件 ----------------------------------------------------
os.system("python3 -m tf2onnx.convert --input %s --output %s --inputs 'x:0' --outputs 'node-2:0' --opset 13" % (pbFile, onnxFile))
print("Succeeded converting model into onnx!")

# 将 .onnx 文件中 TensorRT 不原生支持的节点替换为 Plugin ----------------------------
graph = gs.import_onnx(onnx.load(onnxFile))
graph.inputs[0].shape = ["bs", 3, 4, 5]
graph.outputs[0].shape = ["bs", 3, 4, 5]

for node in graph.nodes:
    if node.op == "Add" and node.name == "node-1":
        scalar = node.inputs[1].values
        pluginV = gs.Variable("MyAddPluginVariable-0", np.dtype(np.float32), None)
        pluginN = gs.Node("AddScalar", "MyAddPluginNode-0", inputs=[node.inputs[0]], outputs=[pluginV], attrs={"scalar": float(scalar)})
        graph.nodes.append(pluginN)
        node.o().inputs[0] = pluginV
        node.outputs.clear()

graph.cleanup()
onnx.save(gs.export_onnx(graph), onnxSurgeonFile)
print("Succeeded inserting AddScalar node!")

# 编译 Plugin 为 .so 文件 --------------------------------------------------------
os.system("make")
print("Succeeded building AddScalar Plugin!")

# TensorRT 中加载 .onnx 和 .so 创建 engine ---------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
ctypes.cdll.LoadLibrary(soFile)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
parser = trt.OnnxParser(network, logger)
with open(onnxFile, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")

inputT0 = network.get_input(0)
inputT0.shape = [-1, nC, nH, nW]
profile.set_shape(inputT0.name, [1, nC, nH, nW], [2, nC, nH, nW], [4, nC, nH, nW])
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
#print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
#for i in range(engine.num_bindings):
#    print("Bind[%2d]:i[%d]->"%(i,i) if engine.binding_is_input(i) else "Bind[%2d]:o[%d]->"%(i,i-nInput),
#        engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i),engine.get_binding_name(i))

bufferH = []
bufferH.append(inputX)
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

printArrayInfomation(inputX, "input")
printArrayInfomation(outputTF, "TF")
printArrayInfomation(bufferH[-1], "TRT")
check(outputTF, bufferH[-1], True)

for buffer in bufferD:
    cudart.cudaFree(buffer)

print("Succeeded running model in TensorRT!")