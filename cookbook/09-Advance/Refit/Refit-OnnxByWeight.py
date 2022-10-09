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
import onnx
import onnx_graphsurgeon as gs
import os
import tensorrt as trt
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable

np.random.seed(31193)
t.manual_seed(97)
t.cuda.manual_seed_all(97)
t.backends.cudnn.deterministic = True
nTrainBatchSize = 128
nInferBatchSize = 1
nHeight = 28
nWidth = 28
onnxFile0 = "./model0.onnx"
onnxFile1 = "./model1.onnx"
weightFile = "para.npz"
trtFile = "./model.plan"
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
inferenceImage = dataPath + "8.png"

os.system("rm -rf ./*.pb ./*.onnx ./*.plan")
np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

# pyTorch 中创建网络--------------------------------------------------------------
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
        return t.from_numpy(data.reshape(1, nHeight, nWidth).astype(np.float32)), t.from_numpy(label)

    def __len__(self):
        return len(self.data)

model = Net().cuda()

# 导出一个未训练的 .onnx 模型
t.onnx.export(model, t.randn(1, 1, nHeight, nWidth, device="cuda"), onnxFile0, input_names=["x"], output_names=["y", "z"], do_constant_folding=True, verbose=True, keep_initializers_as_inputs=True, opset_version=12, dynamic_axes={"x": {0: "nBatchSize"}, "z": {0: "nBatchSize"}})

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

# 导出经训练的 .onnx 模型
t.onnx.export(model, t.randn(1, 1, nHeight, nWidth, device="cuda"), onnxFile1, input_names=["x"], output_names=["y", "z"], do_constant_folding=True, verbose=True, keep_initializers_as_inputs=True, opset_version=12, dynamic_axes={"x": {0: "nBatchSize"}, "z": {0: "nBatchSize"}})
print("Succeeded converting model into ONNX!")

# refit + dynamic shape since TensorRT8.5，这里先把它改成 static shape
for file in [onnxFile0, onnxFile1]:
    graph = gs.import_onnx(onnx.load(file))
    graph.inputs[0].shape = [nInferBatchSize, 1, 28, 28]
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), file)
print("Succeeded converting model into static shape!")

# 从 .onnx 中提取权重 -----------------------------------------------------------
# 先跑一次 trtexec 找出需要提前转置的权重
output = os.popen("trtexec --onnx=%s --refit --buildOnly 2>&1 | grep 'Refitter API,'" % onnxFile1)

nameList = []
permutationList = []
for line in output.readlines():
    print(line)
    name = line.split(" ")[5]
    index0 = line.find("of (") + 4
    index1 = line.find(")! If")
    permutation = line[index0:index1]
    tempList = [int(i) for i in permutation.split(",")]

    nameList.append(name)
    permutationList.append(tempList)

graph = gs.import_onnx(onnx.load(onnxFile1))

# 提取权重保存到 para 中
para = {}
for index, (name, tensor) in enumerate(graph.tensors().items()):
    print("Tensor%4d: name=%s, desc=%s" % (index, name, tensor))
    if str(tensor)[:8] == "Constant":
        if name in nameList:
            print("Weight %s transpose!" % name)
            index = nameList.index(name)
            value = tensor.values.transpose(permutationList[index])
            if value.dtype == np.int64:
                value = value.astype(np.int32)
            para[name] = value
            #para[name] = tensor.values
        else:
            print("Weight %s save!" % name)
            value = tensor.values
            if value.dtype == np.int64:
                value = value.astype(np.int32)
            para[name] = value

np.savez(weightFile, **para)

del para  # 保证后面 TensorRT 部分的 para 是加载 paraFile 得到的，实际使用可以不要这一行

# TensorRT 中加载 .onnx 创建 engine --------------------------------------------
def run():
    logger = trt.Logger(trt.Logger.WARNING)
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            exit()
        print("Succeeded loading engine!")

        para = np.load(weightFile)

        refitter = trt.Refitter(engine, logger)
        layerNameList, weightRoleList = refitter.get_all()
        for name, role in zip(layerNameList, weightRoleList):
            print("LayerName:%s,WeightRole:%s" % (name, role))

        # 更新权重
        # 在 numpy.array -> trt.Weights 的隐式转换时要用 ascontiguousarray 包围
        # trt8.2 和 trt8.4 各节点权重的名字可能不一样，要分别处理
        refitter.set_weights("Conv_0", trt.WeightsRole.KERNEL, np.ascontiguousarray(para["conv1.weight"]))
        refitter.set_weights("Conv_0", trt.WeightsRole.BIAS, np.ascontiguousarray(para["conv1.bias"]))
        refitter.set_weights("Conv_3", trt.WeightsRole.KERNEL, np.ascontiguousarray(para["conv2.weight"]))
        refitter.set_weights("Conv_3", trt.WeightsRole.BIAS, np.ascontiguousarray(para["conv2.bias"]))
        # TRT8.4
        refitter.set_weights("fc1.weight", trt.WeightsRole.CONSTANT, np.ascontiguousarray(para["fc1.weight"]))
        refitter.set_weights("fc1.bias", trt.WeightsRole.CONSTANT, np.ascontiguousarray(para["fc1.bias"]))
        refitter.set_weights("fc2.weight", trt.WeightsRole.CONSTANT, np.ascontiguousarray(para["fc2.weight"]))
        refitter.set_weights("fc2.bias", trt.WeightsRole.CONSTANT, np.ascontiguousarray(para["fc2.bias"]))
        # TRT8.2，部分 Layer 名称与 TRT8.4 不一样
        #refitter.set_weights("Gemm_8", trt.WeightsRole.KERNEL, np.ascontiguousarray(para["fc1.weight"]))
        #refitter.set_weights("Gemm_8", trt.WeightsRole.BIAS, np.ascontiguousarray(para["fc1.bias"]))
        #refitter.set_weights("Gemm_10", trt.WeightsRole.KERNEL, np.ascontiguousarray(para["fc2.weight"]))
        #refitter.set_weights("Gemm_10", trt.WeightsRole.BIAS, np.ascontiguousarray(para["fc2.bias"]))

        refitter.refit_cuda_engine()

    else:
        onnxFile = onnxFile0  # 还没有 model.plan，先用 model0.onnx 构建 model.plan
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.REFIT)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
        #config.max_workspace_size = 3 << 30  # for TRT8.2
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
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            exit()
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    #print("Context binding all? %s" % ("Yes" if context.all_binding_shapes_specified else "No"))
    #for i in range(engine.num_bindings):
    #    print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context.get_binding_shape(i))

    data = cv2.imread(inferenceImage, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    data = np.tile(data, [nInferBatchSize, 1, 1, 1])
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

run()  # 构建 model.plan
run()  # 对 model.plan 做 Refit