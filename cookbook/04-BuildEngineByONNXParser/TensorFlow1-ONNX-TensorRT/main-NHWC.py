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
import tensorflow as tf1
import tensorrt as trt

import calibrator

np.random.seed(31193)
tf1.compat.v1.set_random_seed(97)
nTrainBatchSize = 128
nHeight = 28
nWidth = 28
pbFile = "./model-NHWC.pb"
onnxFile = "./model-NHWC.onnx"
trtFile = "./model-NHWC.plan"
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
inferenceImage = dataPath + "8.png"

trtVersion = trt.__version__.split(".")
# for FP16 mode
bUseFP16Mode = False
# for INT8 model
bUseINT8Mode = False
nCalibration = 1
cacheFile = "./int8.cache"
calibrationDataPath = dataPath + "test/"

os.system("rm -rf ./*.plan ./*.cache")
np.set_printoptions(precision=3, linewidth=100, suppress=True)
tf1.compat.v1.disable_eager_execution()
cudart.cudaDeviceSynchronize()

# Create network and train model in TensorFlow1 --------------------------------
def getBatch(fileList, nSize=1, isTrain=True):
    if isTrain:
        indexList = np.random.choice(len(fileList), nSize)
    else:
        nSize = len(fileList)
        indexList = np.arange(nSize)

    xData = np.zeros([nSize, nHeight, nWidth, 1], dtype=np.float32)
    yData = np.zeros([nSize, 10], dtype=np.float32)
    for i, index in enumerate(indexList):
        imageName = fileList[index]
        data = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
        label = np.zeros(10, dtype=np.float32)
        label[int(imageName[-7])] = 1
        xData[i] = data.reshape(nHeight, nWidth, 1).astype(np.float32) / 255
        yData[i] = label
    return xData, yData

x = tf1.compat.v1.placeholder(tf1.float32, [None, nHeight, nWidth, 1], name="x")
y_ = tf1.compat.v1.placeholder(tf1.float32, [None, 10], name="y_")

w1 = tf1.compat.v1.get_variable("w1", shape=[5, 5, 1, 32], initializer=tf1.truncated_normal_initializer(mean=0, stddev=0.1))
b1 = tf1.compat.v1.get_variable("b1", shape=[32], initializer=tf1.constant_initializer(value=0.1))
h1 = tf1.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding="SAME")
h2 = h1 + b1
h3 = tf1.nn.relu(h2)
h4 = tf1.nn.max_pool2d(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

w2 = tf1.compat.v1.get_variable("w2", shape=[5, 5, 32, 64], initializer=tf1.truncated_normal_initializer(mean=0, stddev=0.1))
b2 = tf1.compat.v1.get_variable("b2", shape=[64], initializer=tf1.constant_initializer(value=0.1))
h5 = tf1.nn.conv2d(h4, w2, strides=[1, 1, 1, 1], padding="SAME")
h6 = h5 + b2
h7 = tf1.nn.relu(h6)
h8 = tf1.nn.max_pool2d(h7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

w3 = tf1.compat.v1.get_variable("w3", shape=[7 * 7 * 64, 1024], initializer=tf1.truncated_normal_initializer(mean=0, stddev=0.1))
b3 = tf1.compat.v1.get_variable("b3", shape=[1024], initializer=tf1.constant_initializer(value=0.1))
h9 = tf1.reshape(h8, [-1, 7 * 7 * 64])
h10 = tf1.matmul(h9, w3)
h11 = h10 + b3
h12 = tf1.nn.relu(h11)

w4 = tf1.compat.v1.get_variable("w4", shape=[1024, 10], initializer=tf1.truncated_normal_initializer(mean=0, stddev=0.1))
b4 = tf1.compat.v1.get_variable("b4", shape=[10], initializer=tf1.constant_initializer(value=0.1))
h13 = tf1.matmul(h12, w4)
h14 = h13 + b4
y = tf1.nn.softmax(h14, name="y")
z = tf1.argmax(y, 1, name="z")

crossEntropy = -tf1.reduce_sum(y_ * tf1.math.log(y))
trainStep = tf1.compat.v1.train.AdamOptimizer(1e-4).minimize(crossEntropy)
accuracy = tf1.reduce_mean(tf1.cast(tf1.equal(z, tf1.argmax(y_, 1)), tf1.float32), name="accuracy")

tfConfig = tf1.compat.v1.ConfigProto()
tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf1.compat.v1.Session(config=tfConfig)
sess.run(tf1.compat.v1.global_variables_initializer())

for i in range(100):
    xSample, ySample = getBatch(trainFileList, nTrainBatchSize, True)
    trainStep.run(session=sess, feed_dict={x: xSample, y_: ySample})
    if i % 10 == 0:
        accuracyValue = accuracy.eval(session=sess, feed_dict={x: xSample, y_: ySample})
        print("%s, batch %3d, acc = %f" % (dt.now(), 10 + i, accuracyValue))

constantGraph = tf1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["z"])
with tf1.gfile.FastGFile(pbFile, mode="wb") as f:
    f.write(constantGraph.SerializeToString())

sess.close()
print("Succeeded building model in TensorFlow1!")

# Export model as ONNX file ----------------------------------------------------
os.system("python3 -m tf2onnx.convert --input %s --output %s --inputs 'x:0' --outputs 'z:0'" % (pbFile, onnxFile))
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