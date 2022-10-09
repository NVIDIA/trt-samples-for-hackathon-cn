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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf2
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorrt as trt

np.random.seed(31193)
tf2.random.set_seed(97)
epsilon = 1e-6
pbFilePath = "./model/"
pbFile = "model.pb"
onnxFile = "./model.onnx"
onnxSurgeonFile = "./model-surgeon.onnx"
soFile = "./AddScalarPlugin.so"
trtFile = "./model.plan"
nB, nC, nH, nW = 2, 3, 4, 5
inputX = np.random.rand(nB, nC, nH, nW).astype(np.float32).reshape([nB, nC, nH, nW])
# 是否保存为单独的一个 .pb文件（两种导出方式），这里选 True 或 Flase 都能导出为 .onnx
isSinglePbFile = True

os.system("rm -rf ./*.pb ./*.onnx ./*.plan ./*.o ./*.d ./*.so")
np.set_printoptions(precision=4, linewidth=200, suppress=True)
tf2.config.experimental.set_memory_growth(tf2.config.list_physical_devices("GPU")[0], True)
cudart.cudaDeviceSynchronize()

def check(a, b, weak=False, info=""):  # 用于比较 TF 和 TRT 的输出结果
    if weak:
        res = np.all(np.abs(a - b) < epsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check %s:" % info, res, diff0, diff1)

def printArrayInfomation(x, info="", n=5):  # 用于输出数组统计信息
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

# TensorFlow 中创建网络并保存为 .pb 文件 -------------------------------------------
modelInput = tf2.keras.Input(shape=[nC, nH, nW], dtype=tf2.dtypes.float32)

layer1 = tf2.keras.layers.Multiply()  # 某些前处理
x = layer1([modelInput, np.array([1], dtype=np.float32)])

layer2 = tf2.keras.layers.Add()  # 想要替换的算子 / 模块
x = layer2([x, np.array([1], dtype=np.float32)])

layer3 = tf2.keras.layers.Multiply()  # 某些后处理
y = layer1([x, np.array([1], dtype=np.float32)])

model = tf2.keras.Model(inputs=modelInput, outputs=y, name="LayerNormExample")

model.summary()

model.compile(
    loss=tf2.keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=tf2.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

outputTF = model(inputX).numpy()

tf2.saved_model.save(model, pbFilePath)

if isSinglePbFile:
    modelFunction = tf2.function(lambda Input: model(Input)).get_concrete_function(tf2.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(modelFunction)
    frozen_func.graph.as_graph_def()
    print("_________________________________________________________________")
    print("Frozen model inputs:\n", frozen_func.inputs)
    print("Frozen model outputs:\n", frozen_func.outputs)
    print("Frozen model layers:")
    for op in frozen_func.graph.get_operations():
        print(op.name)
    tf2.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=pbFilePath, name=pbFile, as_text=False)

print("Succeeded building model in TensorFlow2!")

# 将 .pb 文件转换为 .onnx 文件 ----------------------------------------------------
if isSinglePbFile:
    os.system("python3 -m tf2onnx.convert --input       %s --output %s --opset 13 --inputs 'Input:0' --outputs 'Identity:0'" % (pbFilePath + pbFile, onnxFile))
else:
    os.system("python3 -m tf2onnx.convert --saved-model %s --output %s --opset 13" % (pbFilePath, onnxFile))
print("Succeeded converting model into ONNX!")

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

inputTensor = network.get_input(0)
inputTensor.shape = [-1, nC, nH, nW]
profile.set_shape(inputTensor.name, [1, nC, nH, nW], [2, nC, nH, nW], [4, nC, nH, nW])
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

printArrayInfomation(inputX, "input")
printArrayInfomation(outputTF, "TF")
printArrayInfomation(bufferH[-1], "TRT")
check(outputTF, bufferH[-1], True)

for buffer in bufferD:
    cudart.cudaFree(buffer)

print("Succeeded running model in TensorRT!")