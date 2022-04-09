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

# 用 onnx-graphsurgeon 打印 .onnx 文件逐层信息 ----------------------------------
graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(onnxFile)))

graph.inputs[0].shape = ['B',1,28,28]   # 调整输入维度，字符串表示 dynamic shape
graph.outputs[0].shape = ['B',10]       # 调整输出维度

print("# Traverse the node: -----------------------------------------------------")  # 遍历节点，打印：节点信息，输入张量，输出张量，父节点名，子节点名
for index, node in enumerate(graph.nodes):
    print("Node%4d: op=%s, name=%s, attrs=%s"%(index, node.op,node.name, "".join(["{"] + [str(key)+":"+str(value)+", " for key, value in node.attrs.items()] + ["}"])))
    for jndex, inputTensor in enumerate(node.inputs):
        print("\tInTensor  %d: %s"%(jndex, inputTensor))
    for jndex, outputTensor in enumerate(node.outputs):
        print("\tOutTensor %d: %s"%(jndex, outputTensor))

    fatherNodeList = []
    for newNode in graph.nodes:
        for newOutputTensor in newNode.outputs:
            if newOutputTensor in node.inputs:
                fatherNodeList.append(newNode)
    for jndex, newNode in enumerate(fatherNodeList):
        print("\tFatherNode%d: %s"%(jndex,newNode.name))

    sonNodeList = []
    for newNode in graph.nodes:
        for newInputTensor in newNode.inputs:
            if newInputTensor in node.outputs:
                sonNodeList.append(newNode)
    for jndex, newNode in enumerate(sonNodeList):
        print("\tSonNode   %d: %s"%(jndex,newNode.name))

print("# Traverse the tensor: ---------------------------------------------------") # 遍历张量，打印：张量信息，以本张量作为输入张量的节点名，以本张量作为输出张量的节点名，父张量信息，子张量信息
for index,(name,tensor) in enumerate(graph.tensors().items()):
    print("Tensor%4d: name=%s, desc=%s"%(index, name, tensor))
    for jndex, inputNode in enumerate(tensor.inputs):
        print("\tInNode      %d: %s"%(jndex, inputNode.name))
    for jndex, outputNode in enumerate(tensor.outputs):
        print("\tOutNode     %d: %s"%(jndex, outputNode.name))

    fatherTensorList = []
    for newTensor in list(graph.tensors().values()):
        for newOutputNode in newTensor.outputs:
            if newOutputNode in tensor.inputs:
                fatherTensorList.append(newTensor)
    for jndex, newTensor in enumerate(fatherTensorList):
        print("\tFatherTensor%d: %s"%(jndex,newTensor))

    sonTensorList = []
    for newTensor in list(graph.tensors().values()):
        for newInputNode in newTensor.inputs:
            if newInputNode in tensor.outputs:
                sonTensorList.append(newTensor)
    for jndex, newTensor in enumerate(sonTensorList):
        print("\tSonTensor   %d: %s"%(jndex,newTensor))

