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

from cuda import cudart
import cv2
from datetime import datetime as dt
from glob import glob
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf2
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorrt as trt

import calibrator

np.random.seed(31193)
tf2.random.set_seed(97)
nTrainBatchSize = 128
nHeight = 28
nWidth = 28
pbFilePath = "./model-NHWC-C2/"
pbFile = "model-NHWC-C2.pb"
onnxFile = "./model-NHWC-C2.onnx"
trtFile = "./model-NHWC-C2.plan"
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
inferenceImage = dataPath + "8.png"

# 是否保存为单独的一个 .pb文件（两种导出方式），这里选 True 或 Flase 都能导出为 .onnx
isSinglePbFile = True
# for FP16 mode
bUseFP16Mode = False
# for INT8 model
bUseINT8Mode = False
nCalibration = 1
cacheFile = "./int8.cache"
calibrationDataPath = dataPath + "test/"

os.system("rm -rf %s ./*.plan ./*.cache" % pbFilePath)
np.set_printoptions(precision=4, linewidth=200, suppress=True)
tf2.config.experimental.set_memory_growth(tf2.config.list_physical_devices("GPU")[0], True)
cudart.cudaDeviceSynchronize()

def getData(fileList):
    nSize = len(fileList)
    xData = np.zeros([nSize, nHeight, nWidth, 1], dtype=np.float32)
    yData = np.zeros([nSize, 10], dtype=np.float32)
    for i in range(nSize):
        imageName = fileList[i]
        data = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
        label = np.zeros(10, dtype=np.float32)
        label[int(imageName[-7])] = 1
        xData[i] = data.reshape(nHeight, nWidth, 1).astype(np.float32) / 255
        yData[i] = label
    return xData, yData

# TensorFlow 中创建网络并保存为 .pb 文件 -------------------------------------------
modelInput = tf2.keras.Input(shape=[nHeight, nWidth, 2], dtype=tf2.dtypes.float32)

layerConv1 = tf2.keras.layers.Conv2D(32, [5, 5], strides=[1, 1], padding="same", data_format=None, dilation_rate=[1, 1], groups=1, activation="relu", use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="conv1")
x = layerConv1(modelInput)

layerPool1 = tf2.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="same", data_format=None, name="pool1")
x = layerPool1(x)

layerConv2 = tf2.keras.layers.Conv2D(64, [5, 5], strides=[1, 1], padding="same", data_format=None, dilation_rate=[1, 1], groups=1, activation="relu", use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="conv2")
x = layerConv2(x)

laerPool2 = tf2.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="same", data_format=None, name="pool2")
x = laerPool2(x)

layerReshape = tf2.keras.layers.Reshape([-1], name="reshape")
x = layerReshape(x)

layerDense1 = tf2.keras.layers.Dense(1024, activation="relu", use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="dense1")
x = layerDense1(x)

layerDense2 = tf2.keras.layers.Dense(10, activation=None, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="dense2")
x = layerDense2(x)

layerSoftmax = tf2.keras.layers.Softmax(axis=1, name="softmax")
z = layerSoftmax(x)

model = tf2.keras.Model(inputs=modelInput, outputs=z, name="MNISTExample")

model.summary()

model.compile(
    loss=tf2.keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=tf2.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

xTrain, yTrain = getData(trainFileList)
xTrain = np.tile(xTrain, [1, 1, 1, 2])
history = model.fit(xTrain, yTrain, batch_size=128, epochs=10, validation_split=0.1)

xTest, yTest = getData(testFileList)
xTest = np.tile(xTest, [1, 1, 1, 2])
testScore = model.evaluate(xTest, yTest, verbose=2)
print("%s, loss = %f, accuracy = %f" % (dt.now(), testScore[0], testScore[1]))

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

# TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
if bUseFP16Mode:
    config.set_flag(trt.BuilderFlag.FP16)
if bUseINT8Mode:
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, nHeight, nWidth), cacheFile)
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
inputTensor.shape = [-1, nHeight, nWidth, 2]
profile.set_shape(inputTensor.name, (1, nHeight, nWidth, 2), (4, nHeight, nWidth, 2), (8, nHeight, nWidth, 2))
config.add_optimization_profile(profile)

outputTensor = network.get_output(0)
network.unmark_output(outputTensor)

_17 = network.add_topk(outputTensor, trt.TopKOperation.MAX, 1, 1 << 1)  # 手工补上最后的 ArgMax

network.mark_output(_17.get_output(1))

engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, "wb") as f:
    f.write(engineString)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

context = engine.create_execution_context()
context.set_binding_shape(0, [1, nHeight, nWidth, 2])
#print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
#for i in range(nInput):
#    print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
#for i in range(nInput, nInput + nOutput):
#    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

data = cv2.imread(inferenceImage, cv2.IMREAD_GRAYSCALE).astype(np.float32).reshape(1, nHeight, nWidth, 1)
data = np.tile(data, [1, 1, 1, 2])
bufferH = []
bufferH.append(data)
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

print("inputH0 :", bufferH[0].shape)
print("outputH0:", bufferH[-1].shape)
print(bufferH[-1])
for buffer in bufferD:
    cudart.cudaFree(buffer)
print("Succeeded running model in TensorRT!")