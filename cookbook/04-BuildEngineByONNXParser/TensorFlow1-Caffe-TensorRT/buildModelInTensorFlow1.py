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

#from cuda import cudart
import cv2
from datetime import datetime as dt
from glob import glob
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf1

np.random.seed(31193)
tf1.compat.v1.set_random_seed(97)
nTrainBatchSize = 128
nHeight = 28
nWidth = 28
ckptFile = "./model.ckpt"
pbFile = "./model.pb"
caffeFile = "./model"
trtFile = "./model.plan"
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
inferenceImage = dataPath + "8.png"

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
#h2 = h1 + b1  # Conversion will fail if using bias, see detailed information in result-withBias.txt
h2 = h1
h3 = tf1.nn.relu(h2)
h4 = tf1.nn.max_pool2d(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

w2 = tf1.compat.v1.get_variable("w2", shape=[5, 5, 32, 64], initializer=tf1.truncated_normal_initializer(mean=0, stddev=0.1))
b2 = tf1.compat.v1.get_variable("b2", shape=[64], initializer=tf1.constant_initializer(value=0.1))
h5 = tf1.nn.conv2d(h4, w2, strides=[1, 1, 1, 1], padding="SAME")
#h6 = h5 + b2
h6 = h5
h7 = tf1.nn.relu(h6)
h8 = tf1.nn.max_pool2d(h7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

w3 = tf1.compat.v1.get_variable("w3", shape=[7 * 7 * 64, 1024], initializer=tf1.truncated_normal_initializer(mean=0, stddev=0.1))
b3 = tf1.compat.v1.get_variable("b3", shape=[1024], initializer=tf1.constant_initializer(value=0.1))
h9 = tf1.reshape(h8, [-1, 7 * 7 * 64])
h10 = tf1.matmul(h9, w3)
#h11 = h10 + b3
h11 = h10
h12 = tf1.nn.relu(h11)

w4 = tf1.compat.v1.get_variable("w4", shape=[1024, 10], initializer=tf1.truncated_normal_initializer(mean=0, stddev=0.1))
b4 = tf1.compat.v1.get_variable("b4", shape=[10], initializer=tf1.constant_initializer(value=0.1))
h13 = tf1.matmul(h12, w4)
#h14 = h13 + b4
h14 = h13
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

if True:  # here we use .ckpt to convert the model (（.pb is also OK but the command of mmdnn should be edited).
    saver = tf1.compat.v1.train.Saver(max_to_keep=1)
    saver.save(sess, ckptFile)
else:
    constantGraph = tf1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["y"])
    with tf1.gfile.FastGFile(pbFile, mode="wb") as f:
        f.write(constantGraph.SerializeToString())

sess.close()
print("Succeeded building model in TensorFlow1!")
