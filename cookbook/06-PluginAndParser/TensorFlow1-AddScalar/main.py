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

import ctypes
from cuda import cudart
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import os

os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf1
import tensorrt as trt

np.random.seed(31193)
tf1.compat.v1.set_random_seed(97)
tf1.compat.v1.disable_eager_execution()
epsilon = 1e-6
pbFile = "./model.pb"
onnxFile = "./model.onnx"
onnxSurgeonFile = "./model-surgeon.onnx"
soFile = "./AddScalarPlugin.so"
trtFile = "./model.plan"
nB, nC, nH, nW = 2, 3, 4, 5
inputX = np.random.rand(nB, nC, nH, nW).astype(np.float32).reshape([nB, nC, nH, nW])

np.set_printoptions(precision=3, linewidth=100, suppress=True)
tf1.compat.v1.disable_eager_execution()
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
    print("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))

# Create network in TensorFlow and export as ONNX ----------------------------------
x = tf1.compat.v1.placeholder(tf1.float32, [None, nC, nH, nW], name="x")
_h1 = tf1.multiply(x, 1, name="node-0")  # some preprocess
_h2 = tf1.add(_h1, 1, name="node-1")  # target ndoes
y = tf1.multiply(_h2, 1, name="node-2")  # some postprocess

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

os.system("python3 -m tf2onnx.convert --input %s --output %s --inputs 'x:0' --outputs 'node-2:0' --opset 13" % (pbFile, onnxFile))
print("Succeeded converting model into ONNX!")

# Replace LayerNorm module into LayerNorm plugin node --------------------------
graph = gs.import_onnx(onnx.load(onnxFile))
graph.inputs[0].shape = ["nBS", nC, nH, nW]
graph.outputs[0].shape = ["nBS", nC, nH, nW]

nPlugin = 0
for node in graph.nodes:
    if node.op == "Add":
        scalar = float(node.i(1).attrs["value"].values)
        pluginV = gs.Variable("MyAddPluginVariable-%d" % nPlugin, np.dtype(np.float32), None)
        pluginN = gs.Node("AddScalar", "MyAddPluginNode-%d" % nPlugin, inputs=[node.inputs[0]], outputs=[pluginV], attrs={"scalar": float(scalar)})
        graph.nodes.append(pluginN)
        node.o().inputs[0] = pluginV
        node.outputs.clear()

        nPlugin += 1

graph.cleanup()
onnx.save(gs.export_onnx(graph), onnxSurgeonFile)
print("Succeeded replacing AddScalar plugin!")

# compile plugin.so ------------------------------------------------------------
#os.system("make")  # we do this in the steps in Makefile
#print("Succeeded building LayerNorm Plugin!")

# build TensorRT engine with ONNX file and plugin.so ---------------------------
logger = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(logger, '')
ctypes.cdll.LoadLibrary(soFile)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
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
inputTensor.shape = [-1, nC, nH, nW]
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
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
context.set_input_shape(lTensorName[0], [nB, nC, nH, nW])
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
bufferH.append(np.ascontiguousarray(inputX))
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

check(bufferH[nInput:][0], outputPyTorch, True)

for b in bufferD:
    cudart.cudaFree(b)

print("Succeeded running model in TensorRT!")
