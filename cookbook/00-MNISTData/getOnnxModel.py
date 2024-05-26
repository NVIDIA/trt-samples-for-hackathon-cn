#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
from datetime import datetime as dt
from glob import glob

import cv2
import numpy as np
import onnx
import torch as t
import torch.nn.functional as F
from cuda import cudart
from torch.autograd import Variable

np.random.seed(31193)
t.manual_seed(97)
t.cuda.manual_seed_all(97)
t.backends.cudnn.deterministic = True
batch_size, height, width = 128, 28, 28
dataPath = "/trtcookbook/00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
onnx_file_untrained = "./model0.onnx"
weight_file_untrained = "./model0.npz"
onnx_file_trained = "./model1.onnx"
weight_file_trained = "./model1.npz"
onnx_file_trained_no_weight = "./model2.onnx"
onnx_file_weight = onnx_file_trained_no_weight + ".weight"

os.system("rm -rf ./*.onnx ./*.trt")
np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

# Create network in pyTorch
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
        x = x.reshape(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        z = F.softmax(y, dim=1)
        z = t.argmax(z, dim=1)
        return y, z

class MyData(t.utils.data.Dataset):

    def __init__(self, isTrain=True):
        if isTrain:
            self.data = trainFileList
        else:
            self.data = testFileList

    def __getitem__(self, index):
        imageName = self.data[index]
        data = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
        label = np.zeros(10, dtype=np.float32)
        index = int(imageName[-7])
        label[index] = 1
        return t.from_numpy(data.reshape(1, height, width).astype(np.float32)), t.from_numpy(label)

    def __len__(self):
        return len(self.data)

model = Net().cuda()

# Export untrained model as ONNX file and weight file
t.onnx.export( \
    model,
    t.randn(1, 1, height, width, device="cuda"),
    onnx_file_untrained,
    input_names=["x"],
    output_names=["y", "z"],
    do_constant_folding=True,
    verbose=False,
    keep_initializers_as_inputs=True,
    opset_version=18,
    dynamic_axes={"x": {0: "nBatchSize"}, "z": {0: "nBatchSize"}})

weight = {}
for name, data in model.named_parameters():
    weight[name] = data.detach().cpu().numpy()
np.savez(weight_file_untrained, **weight)

# Train the model
ceLoss = t.nn.CrossEntropyLoss()
opt = t.optim.Adam(model.parameters(), lr=0.001)
trainDataset = MyData(True)
testDataset = MyData(False)
trainLoader = t.utils.data.DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True)
testLoader = t.utils.data.DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True)

for epoch in range(10):
    for xTrain, yTrain in trainLoader:
        xTrain = Variable(xTrain).cuda()
        yTrain = Variable(yTrain).cuda()
        opt.zero_grad()
        y_, z = model(xTrain)
        loss = ceLoss(y_, yTrain)
        loss.backward()
        opt.step()

    with t.no_grad():
        acc = 0
        n = 0
        for xTest, yTest in testLoader:
            xTest = Variable(xTest).cuda()
            yTest = Variable(yTest).cuda()
            y_, z = model(xTest)
            acc += t.sum(z == t.matmul(yTest, t.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to("cuda:0"))).cpu().numpy()
            n += xTest.shape[0]
        print("%s, epoch %2d, loss = %f, test acc = %f" % (dt.now(), epoch + 1, loss.data, acc / n))

print("Succeed building model in pyTorch")

# Export trained model as ONNX file and weight file
t.onnx.export( \
    model,
    t.randn(1, 1, height, width, device="cuda"),
    onnx_file_trained,
    input_names=["x"],
    output_names=["y", "z"],
    do_constant_folding=True,
    verbose=False,
    keep_initializers_as_inputs=True,
    opset_version=18,
    dynamic_axes={"x": {0: "nBatchSize"}, "z": {0: "nBatchSize"}})

# Save a ONNX file with external weight
onnx_model = onnx.load(onnx_file_trained, load_external_data=False)
onnx.save(onnx_model, onnx_file_trained_no_weight, save_as_external_data=True, all_tensors_to_one_file=True, location=onnx_file_weight)

weight = {}
for name, data in model.named_parameters():
    weight[name] = data.detach().cpu().numpy()
np.savez(weight_file_trained, **weight)

print("Finish")
