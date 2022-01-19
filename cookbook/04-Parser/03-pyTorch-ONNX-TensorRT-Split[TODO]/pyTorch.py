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

import numpy as np
from datetime import datetime as dt
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain
import loadMnistData

dataPath    = "./mnistData/"
batchSize   = 100
np.random.seed(97)

class Net1(t.nn.Module):
    def __init__(self):
        super(Net1,self).__init__()
        self.conv1 = t.nn.Conv2d( 1, 32, (5,5), padding=(2,2), bias=True)
        self.conv2 = t.nn.Conv2d(32, 64, (5,5), padding=(2,2), bias=True)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.reshape(-1,7*7*64)
        return x
        
class Net2(t.nn.Module):
    def __init__(self):
        super(Net2,self).__init__()        
        self.fc1 = t.nn.Linear(64*7*7, 1024, bias=True)
        self.fc2 = t.nn.Linear(  1024,   10, bias=True)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

net1 = Net1().cuda()
net2 = Net2().cuda()
ce = t.nn.CrossEntropyLoss()
opt = t.optim.Adam(chain(net1.parameters(),net2.parameters()),lr=0.0001)
mnist = loadMnistData.MnistData(dataPath, isOneHot=False)

for i in range(3000):    
    xTrain, yTrain = mnist.getBatch(batchSize, True)
    xTrain = Variable(t.Tensor(xTrain.transpose(0,3,1,2))).cuda()
    yTrain = Variable(t.Tensor(yTrain).type(t.cuda.LongTensor)).cuda()
    opt.zero_grad()
    y1 = net1(xTrain)
    y2 = net2(y1)
    loss = (y2, yTrain)
    loss.backward()
    opt.step()

    if i%100 == 0:
        acc = t.sum(t.argmax(y2,1) == yTrain).type(t.cuda.FloatTensor)/batchSize
        print( "%s, step %d, loss = %f, acc = %f" %(dt.now(), i, loss.data, acc) )
        
print("finish training!")        
        
xTest, yTest = mnist.getBatch(1, False)
y1 = net1(Variable(t.Tensor(xTest.transpose(0,3,1,2))).cuda())
y2 = net2(y1)
print( "test acc = %f"%(np.sum(np.argmax(y2.data.cpu().numpy(),1) == yTest).astype(np.float32)/len(yTest)) )

t.onnx.export(net1, Variable(t.Tensor(xTest.transpose(0,3,1,2))).cuda(), "./model0.onnx",   
              do_constant_folding=True,example_outputs=y1,opset_version=11,keep_initializers_as_inputs=True)
t.onnx.export(net2, y1, "./model1.onnx",   
              do_constant_folding=True,example_outputs=y2,opset_version=11,keep_initializers_as_inputs=True)

print("finish!")
