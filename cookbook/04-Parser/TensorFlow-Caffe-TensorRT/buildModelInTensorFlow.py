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

import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from datetime import datetime as dt

dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
sys.path.append(dataPath)
import loadMnistData

np.random.seed(97)
tf.compat.v1.set_random_seed(97)
nTrainbatchSize = 128
ckptFile = "./model.ckpt"
pbFile = "./model-NCHW.pb"
caffeFile = "./model"
trtFile = "./model-NCHW.plan"
inputImage = dataPath + '8.png'

np.set_printoptions(precision=4, linewidth=200, suppress=True)

# TensorFlow 中创建网络并保存为 .pb 文件 -------------------------------------------
x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1], name='x')
y_ = tf.compat.v1.placeholder(tf.float32, [None, 10], name='y_')

w1 = tf.compat.v1.get_variable('w1', shape=[5, 5, 1, 32], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
b1 = tf.compat.v1.get_variable('b1', shape=[32], initializer=tf.constant_initializer(value=0.1))
h1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
#h2 = h1 + b1 # 添加偏置项 caffe 会有报错，见 result-withBias.txt
h2 = h1
h3 = tf.nn.relu(h2)
h4 = tf.nn.max_pool2d(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w2 = tf.compat.v1.get_variable('w2', shape=[5, 5, 32, 64], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
b2 = tf.compat.v1.get_variable('b2', shape=[64], initializer=tf.constant_initializer(value=0.1))
h5 = tf.nn.conv2d(h4, w2, strides=[1, 1, 1, 1], padding='SAME')
#h6 = h5 + b2
h6 = h5
h7 = tf.nn.relu(h6)
h8 = tf.nn.max_pool2d(h7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w3 = tf.compat.v1.get_variable('w3', shape=[7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
b3 = tf.compat.v1.get_variable('b3', shape=[1024], initializer=tf.constant_initializer(value=0.1))
h9 = tf.reshape(h8, [-1, 7 * 7 * 64])
h10 = tf.matmul(h9, w3)
#h11 = h10 + b3
h11 = h10
h12 = tf.nn.relu(h11)

w4 = tf.compat.v1.get_variable('w4', shape=[1024, 10], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
b4 = tf.compat.v1.get_variable('b4', shape=[10], initializer=tf.constant_initializer(value=0.1))
h13 = tf.matmul(h12, w4)
#h14 = h13 + b4
h14 = h13
y = tf.nn.softmax(h14, name='y')
#z = tf.argmax(y, -1, name='z')  # Caffe 不支持 ArgMax 操作，在 TensorRT 中再添加

crossEntropy = -tf.reduce_sum(y_ * tf.math.log(y))
trainStep = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(crossEntropy)

output = tf.argmax(y, 1)
resultCheck = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(resultCheck, tf.float32), name='acc')

tfConfig = tf.compat.v1.ConfigProto()
tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.compat.v1.Session(config=tfConfig)
sess.run(tf.compat.v1.global_variables_initializer())

mnist = loadMnistData.MnistData(dataPath, isOneHot=True)
for i in range(1000):
    xSample, ySample = mnist.getBatch(nTrainbatchSize, True)
    trainStep.run(session=sess, feed_dict={x: xSample, y_: ySample})
    if i % 100 == 0:
        train_acc = acc.eval(session=sess, feed_dict={x: xSample, y_: ySample})
        print("%s, step %d, acc = %f" % (dt.now(), i, train_acc))

xSample, ySample = mnist.getBatch(100, False)
print("%s, test acc = %f" % (dt.now(), acc.eval(session=sess, feed_dict={x: xSample, y_: ySample})))

if True:  # 这里使用 .ckpt 来转模型（也可以使用 .pb，但是 mmdnn 的命令需要修改）
    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    saver.save(sess, ckptFile)
else:
    constantGraph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['y'])
    with tf.gfile.FastGFile(pbFile, mode='wb') as f:
        f.write(constantGraph.SerializeToString())

sess.close()
print("Succeeded building model in TensorFlow!")
