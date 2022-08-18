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
import pytorch_quantization.nn as qnn
import pytorch_quantization.calib as calib
from pytorch_quantization.tensor_quant import QuantDescriptor
import sys
import tensorrt as trt
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable

np.random.seed(97)
t.manual_seed(97)
t.cuda.manual_seed_all(97)
t.backends.cudnn.deterministic = True
nTrainBatchSize = 128
nCalibrationBatch = 4
nHeight = 28
nWidth = 28
onnxFile = "./model.onnx"
onnxFilePolygraphy = "./model-p.onnx"
trtFile = "./model.plan"
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
inferenceImage = dataPath + "8.png"

# for QAT
calibrator = ["max", "histogram"][1]
percentileList = [99.9, 99.99, 99.999, 99.9999]
quant_desc_input = QuantDescriptor(calib_method=calibrator, axis=None)
qnn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
qnn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
qnn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
quant_desc_weight = QuantDescriptor(calib_method=calibrator, axis=None)
qnn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
qnn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)
qnn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

os.system("rm -rf %s %s %s" % (onnxFile, onnxFilePolygraphy, trtFile))
np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

# pyTorch 中创建网络--------------------------------------------------------------
class Net(t.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = t.nn.Conv2d(1, 32, (5, 5), padding=(2, 2), bias=True) # 换成对应的 Quantize 系列的 API
        self.conv1 = qnn.QuantConv2d(1, 32, (5, 5), padding=(2, 2), bias=True)
        #self.conv2 = t.nn.Conv2d(32, 64, (5, 5), padding=(2, 2), bias=True)
        self.conv2 = qnn.QuantConv2d(32, 64, (5, 5), padding=(2, 2), bias=True)
        #self.fc1 = t.nn.Linear(64 * 7 * 7, 1024, bias=True)
        self.fc1 = qnn.QuantLinear(64 * 7 * 7, 1024, bias=True)
        #self.fc2 = t.nn.Linear(1024, 10, bias=True)
        self.fc2 = qnn.QuantLinear(1024, 10, bias=True)

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
        return t.from_numpy(data.reshape(1, nHeight, nWidth).astype(np.float32)), t.from_numpy(label)

    def __len__(self):
        return len(self.data)

model = Net().cuda()
ceLoss = t.nn.CrossEntropyLoss()
opt = t.optim.Adam(model.parameters(), lr=0.001)
trainDataset = MyData(True)
testDataset = MyData(False)
trainLoader = t.utils.data.DataLoader(dataset=trainDataset, batch_size=nTrainBatchSize, shuffle=True)
testLoader = t.utils.data.DataLoader(dataset=testDataset, batch_size=nTrainBatchSize, shuffle=True)

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

print("Succeeded building model in pyTorch!")

# pyTorch 中进行模型校正 --------------------------------------------------------
with t.no_grad():
    # 开启校正器
    for name, module in model.named_modules():
        if isinstance(module, qnn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (xTrain, yTrain) in enumerate(trainLoader):
        if i >= nCalibrationBatch:
            break
        model(Variable(xTrain).cuda())

    # 关闭校正器
    for name, module in model.named_modules():
        if isinstance(module, qnn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

    def computeArgMax(model, **kwargs):
        for _, module in model.named_modules():
            if isinstance(module, qnn.TensorQuantizer) and module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)

    if calibrator == "max":
        computeArgMax(model, method="max")
        modelName = "./model-max-%d.pth" % (nCalibrationBatch * trainLoader.batch_size)

    else:
        for percentile in percentileList:
            computeArgMax(model, method="percentile")
            modelName = "./model-percentile-%f-%d.pth" % (percentile, nCalibrationBatch * trainLoader.batch_size)

        for method in ["mse", "entropy"]:
            computeArgMax(model, method=method)
            modelName = "./model-%s-%f.pth" % (method, percentile)

    #t.save(model.state_dict(), modelName)
print("Succeeded calibrating model in pyTorch!")

# pyTorch 中进行模型精调 --------------------------------------------------------
model.cuda()

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

print("Succeeded fine tuning model in pyTorch!")

# 导出模型为 .onnx 文件 ---------------------------------------------------------
model.eval()
qnn.TensorQuantizer.use_fb_fake_quant = True
t.onnx.export(model, t.randn(1, 1, nHeight, nWidth, device="cuda"), onnxFile, input_names=["x"], output_names=["y", "z"], do_constant_folding=True, verbose=True, keep_initializers_as_inputs=True, opset_version=13, dynamic_axes={"x": {0: "nBatchSize"}})
print("Succeeded converting model into onnx!")

# TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
os.system("polygraphy surgeon sanitize --fold-constant %s -o %s" % (onnxFile, onnxFilePolygraphy))

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.flags = 1 << int(trt.BuilderFlag.INT8)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxFilePolygraphy):
    print("Failed finding onnx file!")
    exit()
print("Succeeded finding onnx file!")
with open(onnxFilePolygraphy, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")

inputTensor = network.get_input(0)
profile.set_shape(inputTensor.name, (1, 1, nHeight, nWidth), (4, 1, nHeight, nWidth), (8, 1, nHeight, nWidth))
config.add_optimization_profile(profile)

network.unmark_output(network.get_output(0))  # 去掉输出张量 "y"
engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, "wb") as f:
    f.write(engineString)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

context = engine.create_execution_context()
context.set_binding_shape(0, [1, 1, nHeight, nWidth])
#print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
#for i in range(engine.num_bindings):
#    print("Bind[%2d]:i[%d]->"%(i,i) if engine.binding_is_input(i) else "Bind[%2d]:o[%d]->"%(i,i-nInput),
#            engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i),engine.get_binding_name(i))

data = cv2.imread(inferenceImage, cv2.IMREAD_GRAYSCALE).astype(np.float32).reshape(1, 1, nHeight, nWidth)
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