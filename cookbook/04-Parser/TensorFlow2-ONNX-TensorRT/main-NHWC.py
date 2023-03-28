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
pbFilePath = "./model-NHWC/"
pbFile = "model-NHWC.pb"
onnxFile = "./model-NHWC.onnx"
trtFile = "./model-NHWC.plan"
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
inferenceImage = dataPath + "8.png"

# Two equivalent method to export ONNX file, using single .pb file or several files in a directory
bSinglePbFile = True
# for FP16 mode
bUseFP16Mode = False
# for INT8 model
bUseINT8Mode = False
nCalibration = 1
cacheFile = "./int8.cache"
calibrationDataPath = dataPath + "test/"

os.system("rm -rf %s ./*.plan ./*.cache" % pbFilePath)
np.set_printoptions(precision=3, linewidth=100, suppress=True)
tf2.config.experimental.set_memory_growth(tf2.config.list_physical_devices("GPU")[0], True)
cudart.cudaDeviceSynchronize()

# Create network and train model in TensorFlow2 --------------------------------
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

modelInput = tf2.keras.Input(shape=[nHeight, nWidth, 1], dtype=tf2.dtypes.float32)

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
history = model.fit(xTrain, yTrain, batch_size=128, epochs=10, validation_split=0.1)

xTest, yTest = getData(testFileList)
testScore = model.evaluate(xTest, yTest, verbose=2)
print("%s, loss = %f, accuracy = %f" % (dt.now(), testScore[0], testScore[1]))

tf2.saved_model.save(model, pbFilePath)

if bSinglePbFile:
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

# Export model as ONNX file ----------------------------------------------------
if bSinglePbFile:
    os.system("python3 -m tf2onnx.convert --input       %s --output %s --opset 13 --inputs 'Input:0' --outputs 'Identity:0'" % (pbFilePath + pbFile, onnxFile))
else:
    os.system("python3 -m tf2onnx.convert --saved-model %s --output %s --opset 13" % (pbFilePath, onnxFile))
print("Succeeded converting model into ONNX!")

# Parse network, rebuild network and do inference in TensorRT ------------------
logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
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
inputTensor.shape = [-1, nHeight, nWidth, 1]
profile.set_shape(inputTensor.name, [1, nHeight, nWidth, 1], [4, nHeight, nWidth, 1], [8, nHeight, nWidth, 1])
config.add_optimization_profile(profile)

outputTensor = network.get_output(0)
network.unmark_output(outputTensor)

_17 = network.add_topk(outputTensor, trt.TopKOperation.MAX, 1, 1 << 1)  # add last ArgMax node

network.mark_output(_17.get_output(1))

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
context.set_input_shape(lTensorName[0], [1, nHeight, nWidth, 1])
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
for i in range(nIO):
    bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
bufferD = []
for i in range(nIO):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

data = cv2.imread(inferenceImage, cv2.IMREAD_GRAYSCALE).astype(np.float32).reshape(1, nHeight, nWidth, 1)
bufferH[0] = data

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

for i in range(nIO):
    context.set_tensor_address(lTensorName[i], int(bufferD[i]))

context.execute_async_v3(0)

for i in range(nInput, nIO):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nIO):
    print(lTensorName[i])
    print(bufferH[i])

for b in bufferD:
    cudart.cudaFree(b)

print("Succeeded running model in TensorRT!")