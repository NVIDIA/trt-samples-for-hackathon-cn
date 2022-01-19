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
import numpy as np
import tensorflow as tf
from datetime import datetime as dt
import loadMnistData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
dataPath    = "./mnistData/"
batchSize   = 100
np.random.seed(97)

def wInit(shape, name):
    return tf.get_variable(name, shape = shape, initializer = tf.truncated_normal_initializer(mean=0,stddev=0.1))

def bInit(shape, name):
    return tf.get_variable(name, shape = shape, initializer = tf.constant_initializer(value=0.1))

def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x =  tf.placeholder(tf.float32, [None,28,28,1], name='x')
y_ = tf.placeholder(tf.float32, [None,10], name='y_')

w1 = wInit([5, 5, 1, 32], name='w1')
b1 = bInit([32], name='b1')
h1 = tf.nn.relu(conv(x, w1) + b1)
h1Pool = maxPool(h1)                                                        

w2 = wInit([5, 5, 32, 64], name='w2')
b2 = bInit([64], name='b2')
h2 = tf.nn.relu(conv(h1Pool, w2) + b2)
h2Pool = maxPool(h2)

w3 = wInit([7 * 7 * 64, 1024], name='w3')
b3 = bInit([1024], name='b3')                                               
h2Flat = tf.reshape(h2Pool, [-1, 7*7*64])
h3 = tf.nn.relu(tf.matmul(h2Flat, w3) + b3)

w4 = wInit([1024, 10], name='w4')
b4 = bInit([10], name='b4')
y=tf.nn.softmax(tf.matmul(h3, w4) + b4, name='y')

cross_entropy = -tf.reduce_sum(y_*tf.math.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

output = tf.argmax(y,1)                                                     
resultCheck = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(resultCheck, tf.float32), name='acc')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

mnist = loadMnistData.MnistData(dataPath, isOneHot=True)
for i in range(3000):
    xSample, ySample = mnist.getBatch(batchSize,True)
    train_step.run(session = sess, feed_dict={x: xSample, y_: ySample})
    if i%100 == 0:
        train_acc = acc.eval(session = sess, feed_dict={x:xSample, y_: ySample})
        print("%s, step %d, acc = %f"%(dt.now(), i, train_acc))

xSample, ySample = mnist.getBatch(10000,False)
print( "%s, test acc = %f"%(dt.now(), acc.eval(session = sess, feed_dict={x:xSample, y_: ySample})) )

constantGraph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,['y'])
with tf.gfile.FastGFile("./model.pb", mode='wb') as f:
    f.write(constantGraph.SerializeToString())
sess.close()
