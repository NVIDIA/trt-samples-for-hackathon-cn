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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import onnx
import onnx_graphsurgeon as gs
import numpy as np
from copy import deepcopy
from collections import OrderedDict

tf.compat.v1.set_random_seed(97)
pbFile = "./model.pb"
onnxFile = "./model.onnx"
os.system("rm -rf %s %s %s"%(pbFile,onnxFile))

# TensorFlow 中创建网络并保存为 .pb 文件 ----------------------------------------
x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1], name='x')
y_ = tf.compat.v1.placeholder(tf.float32, [None, 10], name='y_')

w1 = tf.compat.v1.get_variable('w1', shape=[5, 5, 1, 32], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
b1 = tf.compat.v1.get_variable('b1', shape=[32], initializer=tf.constant_initializer(value=0.1))
h1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
h2 = h1 + b1
h3 = tf.nn.relu(h2)
h4 = tf.nn.max_pool2d(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w2 = tf.compat.v1.get_variable('w2', shape=[5, 5, 32, 64], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
b2 = tf.compat.v1.get_variable('b2', shape=[64], initializer=tf.constant_initializer(value=0.1))
h5 = tf.nn.conv2d(h4, w2, strides=[1, 1, 1, 1], padding='SAME')
h6 = h5 + b2
h7 = tf.nn.relu(h6)
h8 = tf.nn.max_pool2d(h7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w3 = tf.compat.v1.get_variable('w3', shape=[7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
b3 = tf.compat.v1.get_variable('b3', shape=[1024], initializer=tf.constant_initializer(value=0.1))
h9 = tf.reshape(h8, [-1, 7 * 7 * 64])
h10 = tf.matmul(h9, w3)
h11 = h10 + b3
h12 = tf.nn.relu(h11)

w4 = tf.compat.v1.get_variable('w4', shape=[1024, 10], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
b4 = tf.compat.v1.get_variable('b4', shape=[10], initializer=tf.constant_initializer(value=0.1))
h13 = tf.matmul(h12, w4)
h14 = h13 + b4
y = tf.nn.softmax(h14, name='y')
z = tf.argmax(y, 1, name='z')

tfConfig = tf.compat.v1.ConfigProto()
tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.compat.v1.Session(config=tfConfig)
sess.run(tf.compat.v1.global_variables_initializer())

constantGraph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['z'])
with tf.gfile.FastGFile("./model.pb", mode='wb') as f:
    f.write(constantGraph.SerializeToString())
sess.close()
print("Succeeded building model in TensorFlow!")

# .pb 文件转 .onnx 文件 ---------------------------------------------------------
os.system("python -m tf2onnx.convert --input %s --output %s --inputs 'x:0' --outputs 'z:0' --inputs-as-nchw 'x:0'" % (pbFile, onnxFile))
print("Succeeded converting model into onnx!")

# TensorRT 中解析 .onnx 文件并打印逐层信息 ---------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
if os.path.isfile(trtFile):
    with open(trtFile, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine == None:
        print("Failed loading engine!")
        exit()
    print("Succeeded loading engine!")
else:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing ONNX file!")

    # 打印逐层信息（注意是 .onnx 文件的逐层信息不是 TensorRT 引擎的逐层信息，因为到这里还没有构建 serialozedNetwork）
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        print(i,"%s,in=%d,out=%d,%s"%(str(layer.type)[10:],layer.num_inputs,layer.num_outputs,layer.name))
        for j in range(layer.num_inputs):
            tensor  =layer.get_input(j)
            if tensor == None:
                print("\tInput  %2d:"%j,"None")
            else:
                print("\tInput  %2d:%s,%s,%s"%(j,tensor.shape,str(tensor.dtype)[9:],tensor.name))
        for j in range(layer.num_outputs):
            tensor  =layer.get_output(j)
            if tensor == None:
                print("\tOutput %2d:"%j,"None")
            else:
                print("\tOutput %2d:%s,%s,%s"%(j,tensor.shape,str(tensor.dtype)[9:],tensor.name))

