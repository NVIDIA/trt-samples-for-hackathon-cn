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
import paddle
import paddle.nn.functional as F
import tensorrt as trt

import calibrator

np.random.seed(31193)
paddle.seed(97)
nTrainBatchSize = 128
nHeight = 28
nWidth = 28
paddleFilePath = "./model/model"
paraFile = "./para.npz"
trtFile = "./model.plan"
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
inferenceImage = dataPath + "8.png"

# for FP16 mode
bUseFP16Mode = False
# for INT8 model
bUseINT8Mode = False
nCalibration = 1
cacheFile = "./int8.cache"
calibrationDataPath = dataPath + "test/"

os.system("rm -rf ./*.npz ./*.plan ./*.cache " + paddleFilePath)
np.set_printoptions(precision=3, linewidth=100, suppress=True)
cudart.cudaDeviceSynchronize()

def getBatch(fileList, nSize=1, isTrain=True):
    if isTrain:
        indexList = np.random.choice(len(fileList), nSize)
    else:
        nSize = len(fileList)
        indexList = np.arange(nSize)

    xData = np.zeros([nSize, 1, nHeight, nWidth], dtype=np.float32)
    yData = np.zeros([nSize, 10], dtype=np.float32)
    for i, index in enumerate(indexList):
        imageName = fileList[index]
        data = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
        label = np.zeros(10, dtype=np.float32)
        label[int(imageName[-7])] = 1
        xData[i] = data.reshape(1, nHeight, nWidth).astype(np.float32) / 255
        yData[i] = label
    return xData, yData

# Create network and train model in Paddlepaddle -------------------------------
class Net(paddle.nn.Layer):

    def __init__(self, num_classes=1):
        super(Net, self).__init__()

        self.conv1 = paddle.nn.Conv2D(1, 32, [5, 5], 1, 2)
        self.pool1 = paddle.nn.MaxPool2D(2, 2)
        self.conv2 = paddle.nn.Conv2D(32, 64, [5, 5], 1, 2)
        self.pool2 = paddle.nn.MaxPool2D(2, 2)
        self.flatten = paddle.nn.Flatten(1)
        self.fc1 = paddle.nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = paddle.nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        y = self.fc2(x)
        z = F.softmax(y, 1)
        z = paddle.argmax(z, 1)
        return y, z

model = Net()

model.train()
opt = paddle.optimizer.Adam(0.001, parameters=model.parameters())
for i in range(100):
    xSample, ySample = getBatch(trainFileList, nTrainBatchSize, True)
    xSample = paddle.to_tensor(xSample)
    ySample = paddle.to_tensor(ySample)
    y, z = model(xSample)
    loss = F.cross_entropy(y, paddle.argmax(ySample, 1, keepdim=True))
    loss.backward()
    opt.step()
    opt.clear_grad()

    if i % 10 == 0:
        accuracyValue = paddle.sum(z - paddle.argmax(ySample, 1) == 0).numpy().item() / nTrainBatchSize
        print("%s, batch %3d, train acc = %f" % (dt.now(), 10 + i, accuracyValue))

model.eval()
xTest, yTest = getBatch(testFileList, nTrainBatchSize, False)
xTest = paddle.to_tensor(xTest)
yTest = paddle.to_tensor(yTest)
accuracyValue = 0
for i in range(len(testFileList) // nTrainBatchSize):
    xSample = xTest[i * nTrainBatchSize:(i + 1) * nTrainBatchSize]
    ySample = yTest[i * nTrainBatchSize:(i + 1) * nTrainBatchSize]
    y, z = model(xSample)
    accuracyValue += paddle.sum(z - paddle.argmax(ySample, 1) == 0).numpy().item()
print("%s, test acc = %f" % (dt.now(), accuracyValue / (len(testFileList) // nTrainBatchSize * nTrainBatchSize)))

# Two methods to save weights as file
if True:  # extract weights from model
    print("Parameter of the model:")
    para = {}
    for item in model.named_parameters():
        print(item[0], ":", item[1].shape)
        para[item[0]] = item[1]
    np.savez(paraFile, **para)

else:  # etract weights from file
    inputDescList = []
    inputDescList.append(paddle.static.InputSpec(shape=[None, 1, nHeight, nWidth], dtype='float32', name='x'))
    modelStatic = paddle.jit.to_static(model, inputDescList)
    paddle.jit.save(modelStatic, paddleFilePath)

    stateDict = paddle.load(paddleFilePath)
    print("Parameter of the model:")
    para = {}
    for key in stateDict.keys():
        print(key, ":", stateDict[key].shape)
        para[key] = stateDict[key]
    np.savez(paraFile, **para)

del para
print("Succeeded building model in Paddlepaddle!")

# Rebuild network, load weights and do inference in TensorRT -------------------
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
if bUseFP16Mode:
    config.set_flag(trt.BuilderFlag.FP16)
if bUseINT8Mode:
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, nHeight, nWidth), cacheFile)

inputTensor = network.add_input("inputT0", trt.float32, [-1, 1, nHeight, nWidth])
profile.set_shape(inputTensor.name, [1, 1, nHeight, nWidth], [4, 1, nHeight, nWidth], [8, 1, nHeight, nWidth])
config.add_optimization_profile(profile)

para = np.load(paraFile)

w = np.ascontiguousarray(para["conv1.weight"])
b = np.ascontiguousarray(para["conv1.bias"])
_0 = network.add_convolution_nd(inputTensor, 32, [5, 5], trt.Weights(w), trt.Weights(b))
_0.padding_nd = [2, 2]
_1 = network.add_activation(_0.get_output(0), trt.ActivationType.RELU)
_2 = network.add_pooling_nd(_1.get_output(0), trt.PoolingType.MAX, [2, 2])
_2.stride_nd = [2, 2]

w = np.ascontiguousarray(para["conv2.weight"])
b = np.ascontiguousarray(para["conv2.bias"])
_3 = network.add_convolution_nd(_2.get_output(0), 64, [5, 5], trt.Weights(w), trt.Weights(b))
_3.padding_nd = [2, 2]
_4 = network.add_activation(_3.get_output(0), trt.ActivationType.RELU)
_5 = network.add_pooling_nd(_4.get_output(0), trt.PoolingType.MAX, [2, 2])
_5.stride_nd = [2, 2]

_6 = network.add_shuffle(_5.get_output(0))
_6.reshape_dims = (-1, 64 * 7 * 7)

w = np.ascontiguousarray(para["fc1.weight"])
b = np.ascontiguousarray(para["fc1.bias"].reshape(1, -1))
_7 = network.add_constant(w.shape, trt.Weights(w))
_8 = network.add_matrix_multiply(_6.get_output(0), trt.MatrixOperation.NONE, _7.get_output(0), trt.MatrixOperation.NONE)
_9 = network.add_constant(b.shape, trt.Weights(b))
_10 = network.add_elementwise(_8.get_output(0), _9.get_output(0), trt.ElementWiseOperation.SUM)
_11 = network.add_activation(_10.get_output(0), trt.ActivationType.RELU)

w = np.ascontiguousarray(para["fc2.weight"])
b = np.ascontiguousarray(para["fc2.bias"].reshape(1, -1))
_12 = network.add_constant(w.shape, trt.Weights(w))
_13 = network.add_matrix_multiply(_11.get_output(0), trt.MatrixOperation.NONE, _12.get_output(0), trt.MatrixOperation.NONE)
_14 = network.add_constant(b.shape, trt.Weights(b))
_15 = network.add_elementwise(_13.get_output(0), _14.get_output(0), trt.ElementWiseOperation.SUM)

_16 = network.add_softmax(_15.get_output(0))
_16.axes = 1 << 1

_17 = network.add_topk(_16.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)

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
context.set_input_shape(lTensorName[0], [1, 1, nHeight, nWidth])
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
data = cv2.imread(inferenceImage, cv2.IMREAD_GRAYSCALE).astype(np.float32).reshape(1, 1, nHeight, nWidth)
bufferH.append(np.ascontiguousarray(data))
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

for i in range(nIO):
    print(lTensorName[i])
    print(bufferH[i])

for b in bufferD:
    cudart.cudaFree(b)

print("Succeeded running model in TensorRT!")