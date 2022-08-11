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
import sys
import cv2
import numpy as np
from glob import glob
from datetime import datetime as dt
import torch as t
import torch_tensorrt

from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable

dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
sys.path.append(dataPath)
import loadMnistData

nTrainBatchSize = 128
tsFile = "./model.ts"
inferenceImage = dataPath + "8.png"
nHeight = 28
nWidth = 28

os.system("rm -rf ./*.pt ./*.ps")
t.manual_seed(97)
np.set_printoptions(precision=4, linewidth=200, suppress=True)

# pyTorch 中创建网络 -------------------------------------------------------------
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
        return y  # Torch TensorRT 不支持 argmax，不在网络中计算 softmax 和 argmax

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
        return t.from_numpy(data.reshape(1, nHeight, nWidth).astype(np.float32)), label

    def __len__(self):
        return len(self.data)

net = Net().cuda()
ceLoss = t.nn.CrossEntropyLoss()
opt = t.optim.Adam(net.parameters(), lr=0.001)
trainDataset = MyData(isTrain=True, nTrain=600)
testDataset = MyData(isTrain=False, nTest=100)
trainLoader = t.utils.data.DataLoader(dataset=trainDataset, batch_size=nTrainBatchSize, shuffle=True)
testLoader = t.utils.data.DataLoader(dataset=testDataset, batch_size=nTrainBatchSize, shuffle=True)

for epoch in range(30):
    for i, (xTrain, yTrain) in enumerate(trainLoader):
        xTrain = Variable(xTrain).cuda()
        yTrain = Variable(yTrain).cuda()
        opt.zero_grad()
        y_ = net(xTrain)
        loss = ceLoss(y_, yTrain)
        loss.backward()
        opt.step()
    if not (epoch + 1) % 10:
        print("%s, epoch %d, loss = %f" % (dt.now(), epoch + 1, loss.data))

acc = 0
net.eval()
for xTest, yTest in testLoader:
    xTest = Variable(xTest).cuda()
    yTest = Variable(yTest).cuda()
    y_ = net(xTest)
    acc += t.sum(t.argmax(t.softmax(y_, dim=1), dim=1) == t.matmul(yTest, t.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to("cuda:0"))).cpu().numpy()
print("test acc = %f" % (acc / len(testLoader) / nTrainBatchSize))

# 使用 Torch-TensorRT -----------------------------------------------------------
tsModel = t.jit.trace(net, t.randn(1, 1, nHeight, nWidth, device="cuda"))
trtModel = torch_tensorrt.compile(tsModel, inputs=[t.randn(1, 1, nHeight, nWidth, device="cuda").float()], enabled_precisions={t.float})

data = cv2.imread(inferenceImage, cv2.IMREAD_GRAYSCALE).reshape(1, 1, 28, 28).astype(np.float32)
inputData = t.from_numpy(data).cuda()
outputData = trtModel(inputData)  # run inference in TensorRT
print(t.argmax(t.softmax(outputData, dim=1), dim=1))

t.jit.save(trtModel, tsFile)  # 保存 TRT embedded Torchscript
