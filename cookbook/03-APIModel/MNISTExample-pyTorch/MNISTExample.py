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

from cuda import cudart
import cv2
from datetime import datetime as dt
from glob import glob
import numpy as np
import os
import sys
import tensorrt as trt
import torch as t
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable

import calibrator

dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
sys.path.append(dataPath)
import loadMnistData

np.random.seed(97)
t.manual_seed(97)
t.cuda.manual_seed_all(97)
t.backends.cudnn.deterministic = True
nTrainBatchSize = 128
paraFile = "./para.npz"
trtFile = "./model.plan"
inputImage = dataPath + '8.png'

# for FP16 mode
isFP16Mode = False
# for INT8 model
isINT8Mode = False
calibrationDataPath = dataPath + "test/"
calibrationCount = 1
cacheFile = "./int8.cache"
calibrationDataPath = dataPath + "test/"
imageHeight = 28
imageWidth = 28

os.system("rm -rf ./*.npz ./*.plan ./*.cache")
np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

# pyTorch 中创建网络并保存为 .pt 文件 ----------------------------------------------
class Net(t.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = t.nn.Conv2d(1, 32, (5, 5), padding=(2, 2), bias=True)
        self.conv2 = t.nn.Conv2d(32, 64, (5, 5), padding=(2, 2), bias=True)
        self.fc1 = t.nn.Linear(64 * 7 * 7, 1024, bias=True)
        self.fc2 = t.nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.reshape(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        z = F.softmax(y, dim=1)
        z = t.argmax(z, dim=1)
        return y, z

class MyData(data.Dataset):

    def __init__(self, path=dataPath, isTrain=True, nTrain=0, nTest=0):
        if isTrain:
            if len(glob(dataPath + "train/*.jpg")) == 0:
                mnist = loadMnistData.MnistData(path, isOneHot=False)
                mnist.saveImage([60000, nTrain][int(nTrain > 0)], path + "train/", True)  # 60000 images in total
            self.data = glob(path + "train/*.jpg")
        else:
            if len(glob(dataPath + "test/*.jpg")) == 0:
                mnist = loadMnistData.MnistData(path, isOneHot=False)
                mnist.saveImage([10000, nTest][int(nTest > 0)], path + "test/", False)  # 10000 images in total
            self.data = glob(path + "test/*.jpg")

    def __getitem__(self, index):
        imageName = self.data[index]
        data = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
        label = np.zeros(10, dtype=np.float32)
        index = int(imageName[-7])
        label[index] = 1
        return t.from_numpy(data.reshape(1, imageHeight, imageWidth).astype(np.float32)), label

    def __len__(self):
        return len(self.data)

model = Net().cuda()
ceLoss = t.nn.CrossEntropyLoss()
opt = t.optim.Adam(model.parameters(), lr=0.001)
trainDataset = MyData(isTrain=True, nTrain=600)
testDataset = MyData(isTrain=False, nTest=100)
trainLoader = t.utils.data.DataLoader(dataset=trainDataset, batch_size=nTrainBatchSize, shuffle=True)
testLoader = t.utils.data.DataLoader(dataset=testDataset, batch_size=nTrainBatchSize, shuffle=True)

for epoch in range(40):
    for i, (xTrain, yTrain) in enumerate(trainLoader):
        xTrain = Variable(xTrain).cuda()
        yTrain = Variable(yTrain).cuda()
        opt.zero_grad()
        y_, z = model(xTrain)
        loss = ceLoss(y_, yTrain)
        loss.backward()
        opt.step()
    if not (epoch + 1) % 10:
        print("%s, epoch %d, loss = %f" % (dt.now(), epoch + 1, loss.data))

acc = 0
model.eval()
for xTest, yTest in testLoader:
    xTest = Variable(xTest).cuda()
    yTest = Variable(yTest).cuda()
    y_, z = model(xTest)
    acc += t.sum(z == t.matmul(yTest, t.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to('cuda:0'))).cpu().numpy()
print("test acc = %f" % (acc / len(testLoader) / nTrainBatchSize))

para = {}  # 保存权重
for name, parameter in model.named_parameters():
    #print(name, parameter.detach().cpu().numpy().shape)
    para[name] = parameter.detach().cpu().numpy()
np.savez(paraFile, **para)
print("Succeeded building model in pyTorch!")

# TensorRT 中重建网络并创建 engine ------------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
if os.path.isfile(trtFile):
    with open(trtFile, 'rb') as f:
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
    config.max_workspace_size = 3 << 30
    if isFP16Mode:
        config.flags = 1 << int(trt.BuilderFlag.FP16)
    if isINT8Mode:
        config.flags = 1 << int(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, calibrationCount, (1, 1, 28, 28), cacheFile)

    inputTensor = network.add_input('inputT0', trt.float32, [-1, 1, 28, 28])
    profile.set_shape(inputTensor.name, (1, 1, 28, 28), (4, 1, 28, 28), (8, 1, 28, 28))
    config.add_optimization_profile(profile)

    para = np.load(paraFile)

    w = para['conv1.weight'].reshape(-1)
    b = para['conv1.bias']
    _0 = network.add_convolution_nd(inputTensor, 32, [5, 5], w, b)
    _0.padding_nd = [2, 2]
    _1 = network.add_activation(_0.get_output(0), trt.ActivationType.RELU)
    _2 = network.add_pooling_nd(_1.get_output(0), trt.PoolingType.MAX, [2, 2])
    _2.stride_nd = [2, 2]

    w = para['conv2.weight'].reshape(-1)
    b = para['conv2.bias']
    _3 = network.add_convolution_nd(_2.get_output(0), 64, [5, 5], w, b)
    _3.padding_nd = [2, 2]
    _4 = network.add_activation(_3.get_output(0), trt.ActivationType.RELU)
    _5 = network.add_pooling_nd(_4.get_output(0), trt.PoolingType.MAX, [2, 2])
    _5.stride_nd = [2, 2]

    _6 = network.add_shuffle(_5.get_output(0))
    _6.reshape_dims = (-1, 64 * 7 * 7, 1, 1)

    w = para['fc1.weight'].reshape(-1)
    b = para['fc1.bias']
    _7 = network.add_fully_connected(_6.get_output(0), 1024, w, b)
    _8 = network.add_activation(_7.get_output(0), trt.ActivationType.RELU)

    w = para['fc2.weight'].reshape(-1)
    b = para['fc2.bias']
    _9 = network.add_fully_connected(_8.get_output(0), 10, w, b)

    _10 = network.add_shuffle(_9.get_output(0))
    _10.reshape_dims = [-1, 10]

    _11 = network.add_softmax(_10.get_output(0))
    _11.axes = 1 << 1

    _12 = network.add_topk(_11.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)

    network.mark_output(_12.get_output(1))

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write(engineString)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

context = engine.create_execution_context()
context.set_binding_shape(0, [1, 1, 28, 28])
_, stream = cudart.cudaStreamCreate()
print("Binding0->", engine.get_binding_shape(0), context.get_binding_shape(0), engine.get_binding_dtype(0))
print("Binding1->", engine.get_binding_shape(1), context.get_binding_shape(1), engine.get_binding_dtype(1))

data = cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE).astype(np.float32).reshape(1, 1, 28, 28)
inputH0 = np.ascontiguousarray(data.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
_, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
cudart.cudaStreamSynchronize(stream)

print("inputH0 :", data.shape)
#print(data)
print("outputH0:", outputH0.shape)
print(outputH0)

cudart.cudaStreamDestroy(stream)
cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
print("Succeeded running model in TensorRT!")
