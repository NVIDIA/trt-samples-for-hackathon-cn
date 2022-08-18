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
import tensorflow as tf1
import tensorrt as trt
import uff

np.random.seed(97)
tf1.compat.v1.set_random_seed(97)
nTrainBatchSize = 128
nHeight = 28
nWidth = 28
pbFile = "./model.pb"
uffFile = "./model.uff"
trtFile = "./model.plan"
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
inferenceImage = dataPath + "8.png"

os.system("rm -rf ./*.plan ./*.cache")
np.set_printoptions(precision=4, linewidth=200, suppress=True)
tf1.compat.v1.disable_eager_execution()
cudart.cudaDeviceSynchronize()

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

# TensorFlow 中创建网络并保存为 .pb 文件 -------------------------------------------
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

# 将 .pb 文件转换为 .uff 文件 -----------------------------------------------------
uff.from_tensorflow_frozen_model(
    pbFile,
    output_nodes=["y"],
    output_filename=uffFile,
    preprocessor=None,
    write_preprocessed=False,
    text=False,
    quiet=False,
    debug_mode=False,
    #input_node=["x"],
    #list_nodes=False,
    return_graph_info=False
)
print("Succeeded converting model into .uff!")

# TensorRT 中加载 .uff 创建 engine -----------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()  # 使用 implicit batch 模式
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)

parser = trt.UffParser()
parser.register_input("x", [28, 28, 1], trt.UffInputOrder.NHWC)
parser.register_output("y")
parser.parse(uffFile, network)

engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, "wb") as f:
    f.write(engineString)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

context = engine.create_execution_context()
#print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
#for i in range(engine.num_bindings):
#    print("Bind[%2d]:i[%d]->"%(i,i) if engine.binding_is_input(i) else "Bind[%2d]:o[%d]->"%(i,i-nInput),
#            engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i),engine.get_binding_name(i))

data = cv2.imread(inferenceImage, cv2.IMREAD_GRAYSCALE).astype(np.float32).reshape(1, nHeight, nWidth, 1)
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
