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
import onnx
import onnx_graphsurgeon as gs
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2, meta_graph_pb2, rewriter_config_pb2
from tensorflow.python.framework import importer, ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver
import tensorrt as trt

dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
sys.path.append(dataPath)
import loadMnistData

tf.compat.v1.disable_eager_execution()
np.random.seed(31193)
tf.compat.v1.set_random_seed(97)
nTrainBatchSize = 128
nHeight = 28
nWidth = 28
ckptFile = "./model/model.ckpt"
pbFile = "model-V1.pb"
pb2File = "model-V2.pb"
onnxFile = "model-V1.onnx"
onnx2File = "model-V2.onnx"
trtFile = "model.plan"
inferenceImage = dataPath + "8.png"
outputNodeName = "z"
isRemoveTransposeNode = False
isAddQDQForInput = False

os.system("rm -rf ./model*.* checkpoint")

# Create network and train model in TensorFlow1 --------------------------------
g1 = tf.Graph()
with g1.as_default():
    x = tf.compat.v1.placeholder(tf.float32, [None, 1, 28, 28], name="input_0")
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10], name="output_0")

    h0 = tf.transpose(x, [0, 2, 3, 1])

    w1 = tf.compat.v1.get_variable("w1", shape=[5, 5, 1, 32], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    b1 = tf.compat.v1.get_variable("b1", shape=[32], initializer=tf.constant_initializer(value=0.1))
    h1 = tf.nn.conv2d(h0, w1, strides=[1, 1, 1, 1], padding="SAME")
    h2 = h1 + b1
    h3 = tf.nn.relu(h2)
    h4 = tf.nn.max_pool2d(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    w2 = tf.compat.v1.get_variable("w2", shape=[5, 5, 32, 64], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    b2 = tf.compat.v1.get_variable("b2", shape=[64], initializer=tf.constant_initializer(value=0.1))
    h5 = tf.nn.conv2d(h4, w2, strides=[1, 1, 1, 1], padding="SAME")
    h6 = h5 + b2
    h7 = tf.nn.relu(h6)
    h8 = tf.nn.max_pool2d(h7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    w3 = tf.compat.v1.get_variable("w3", shape=[7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    b3 = tf.compat.v1.get_variable("b3", shape=[1024], initializer=tf.constant_initializer(value=0.1))
    h9 = tf.reshape(h8, [-1, 7 * 7 * 64])
    h10 = tf.matmul(h9, w3)
    h11 = h10 + b3
    h12 = tf.nn.relu(h11)

    w4 = tf.compat.v1.get_variable("w4", shape=[1024, 10], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    b4 = tf.compat.v1.get_variable("b4", shape=[10], initializer=tf.constant_initializer(value=0.1))
    h13 = tf.matmul(h12, w4)
    h14 = h13 + b4

    loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=y_, logits=h14)
    acc = tf.compat.v1.metrics.accuracy(labels=tf.argmax(y_, axis=1), predictions=tf.argmax(h14, axis=1))[1]

    tf.contrib.quantize.experimental_create_training_graph(tf.compat.v1.get_default_graph(), symmetric=True, use_qdq=True, quant_delay=800)

    trainStep = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)

mnist = loadMnistData.MnistData(dataPath, isOneHot=True)
with tf.Session(graph=g1) as sess:
    sess.run(tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()))
    for i in range(1000):
        xSample, ySample = mnist.getBatch(nTrainBatchSize, True)
        trainStep.run(session=sess, feed_dict={x: xSample.reshape(-1, 1, 28, 28), y_: ySample})
        if (i % 100 == 0):
            xSample, ySample = mnist.getBatch(100, False)
            test_accuracy = sess.run(acc, {x: xSample.reshape(-1, 1, 28, 28), y_: ySample})
            print("%s, Step=%d, test acc=%.3f" % (dt.now(), i, acc.eval(session=sess, feed_dict={x: xSample.reshape(-1, 1, 28, 28), y_: ySample})))
    tf.compat.v1.train.Saver().save(sess, ckptFile)

    xSample, ySample = mnist.getBatch(100, False)
    print("%s, test acc = %.3f" % (dt.now(), acc.eval(session=sess, feed_dict={x: xSample.reshape(-1, 1, 28, 28), y_: ySample})))
print("Succeeded saving .ckpt in TensorFlow!")

# Save network as .pb ----------------------------------------------------------
g2 = tf.Graph()
with g2.as_default():
    x = tf.compat.v1.placeholder(tf.float32, [None, 1, 28, 28], name="input_0")

    h0 = tf.transpose(x, [0, 2, 3, 1])

    w1 = tf.compat.v1.get_variable("w1", shape=[5, 5, 1, 32], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    b1 = tf.compat.v1.get_variable("b1", shape=[32], initializer=tf.constant_initializer(value=0.1))
    h1 = tf.nn.conv2d(h0, w1, strides=[1, 1, 1, 1], padding="SAME")
    h2 = h1 + b1
    h3 = tf.nn.relu(h2)
    h4 = tf.nn.max_pool2d(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    w2 = tf.compat.v1.get_variable("w2", shape=[5, 5, 32, 64], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    b2 = tf.compat.v1.get_variable("b2", shape=[64], initializer=tf.constant_initializer(value=0.1))
    h5 = tf.nn.conv2d(h4, w2, strides=[1, 1, 1, 1], padding="SAME")
    h6 = h5 + b2
    h7 = tf.nn.relu(h6)
    h8 = tf.nn.max_pool2d(h7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    w3 = tf.compat.v1.get_variable("w3", shape=[7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    b3 = tf.compat.v1.get_variable("b3", shape=[1024], initializer=tf.constant_initializer(value=0.1))
    h9 = tf.reshape(h8, [-1, 7 * 7 * 64])
    h10 = tf.matmul(h9, w3)
    h11 = h10 + b3
    h12 = tf.nn.relu(h11)

    w4 = tf.compat.v1.get_variable("w4", shape=[1024, 10], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    b4 = tf.compat.v1.get_variable("b4", shape=[10], initializer=tf.constant_initializer(value=0.1))
    h13 = tf.matmul(h12, w4)
    h14 = h13 + b4

    h15 = tf.nn.softmax(h14, name="softmax", axis=1)
    h16 = tf.argmax(h15, 1, name=outputNodeName)

    tf.contrib.quantize.experimental_create_eval_graph(symmetric=True, use_qdq=True)

with tf.Session(graph=g2) as sess:
    tf.compat.v1.train.Saver().restore(sess, ckptFile)
    constantGraph = tf.graph_util.convert_variables_to_constants(sess, g2.as_graph_def(), [outputNodeName])
with tf.gfile.FastGFile(pbFile, mode="wb") as f:
    f.write(constantGraph.SerializeToString())
print("Succeeded saving .pb in TensorFlow!")

# Optimize .pb -----------------------------------------------------------------
with open(pbFile, "rb") as f:
    graphdef = graph_pb2.GraphDef()
    graphdef.ParseFromString(f.read())

graph = ops.Graph()
with graph.as_default():
    outputCollection = meta_graph_pb2.CollectionDef()
    for output in outputNodeName.split(","):
        outputCollection.node_list.value.append(output)
    importer.import_graph_def(graphdef, name="")
    metagraph = saver.export_meta_graph(graph_def=graph.as_graph_def(add_shapes=True), graph=graph)
    metagraph.collection_def["train_op"].CopyFrom(outputCollection)

rewriter_config = rewriter_config_pb2.RewriterConfig()
rewriter_config.optimizers.extend(["constfold"])
rewriter_config.meta_optimizer_iterations = (rewriter_config_pb2.RewriterConfig.ONE)
session_config = config_pb2.ConfigProto()
session_config.graph_options.rewrite_options.CopyFrom(rewriter_config)

folded_graph = tf_optimizer.OptimizeGraph(session_config, metagraph)

with open(pb2File, "wb") as f:
    f.write(folded_graph.SerializeToString())
print("Succeeded optimizing .pb in TensorFlow!")

# Export model as ONNX file ----------------------------------------------------
os.system("python3 -m tf2onnx.convert --opset 11 --input %s --output %s --inputs 'input_0:0' --outputs '%s:0' --inputs-as-nchw 'x:0'" % (pb2File, onnxFile, outputNodeName))
print("Succeeded converting model into ONNX!")

# Optimize ONNX file to remove Transpose node before Conv node -----------------
graph = gs.import_onnx(onnx.load(onnxFile))

# In the original repo there is a Transpose node before the weight input of the Conv node, which is not supported by the TensorRT QAT mode, so we need to transpose the weight manually and remove the Transpose node.
# However, this Transpose node does not appear in the current exported ONNX file, so this step is no longer needed.
if isRemoveTransposeNode:
    for node in [n for n in graph.nodes if n.op == "Conv"]:
        convKernelTensor = node.i(1).i().i().inputs[0]
        convKernelTensor.values = convKernelTensor.values.transpose(3, 2, 0, 1)
        node.inputs[1] = node.i(1).i(0).outputs[0]

onnx.save_model(gs.export_onnx(graph.cleanup().toposort()), onnx2File)
print("Succeeded optimizing .onnx in Onnx!")

# Parse ONNX file, rebuild network and do inference in TensorRT ----------------
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
networkFlag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION))
network = builder.create_network(networkFlag)
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
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
inputTensor.shape = [-1, 1, nHeight, nWidth]
profile.set_shape(inputTensor.name, [1, 1, nHeight, nWidth], [4, 1, nHeight, nWidth], [8, 1, nHeight, nWidth])
config.add_optimization_profile(profile)

# Add the Quantize/Dequantize node for all input tensors, here we use empirical value 127<->1.0.
# Ideally, these values need to be obtained in the QAT process.
if isAddQDQForInput:
    quantizeScale = np.array([1.0 / 127.0], dtype=np.float32)
    dequantizeScale = np.array([127.0 / 1.0], dtype=np.float32)
    one = np.array([1], dtype=np.float32)
    zero = np.array([0], dtype=np.float32)

    for i in range(network.num_inputs):
        inputTensor = network.get_input(i)

        for j in range(network.num_layers):
            layer = network.get_layer(j)

            for k in range(layer.num_inputs):
                if (layer.get_input(k) == inputTensor):
                    print(i, layer, k)
                    #quantizeLayer = network.add_scale(inputTensor, trt.ScaleMode.UNIFORM, zero, quantizeScale)
                    quantizeLayer = network.add_scale(inputTensor, trt.ScaleMode.UNIFORM, zero, one)
                    quantizeLayer.set_output_type(0, trt.int8)
                    quantizeLayer.name = "InputQuantizeNode"
                    quantizeLayer.get_output(0).name = "QuantizedInputTensor"
                    #dequantizeLayer = network.add_scale(quantizeLayer.get_output(0), trt.ScaleMode.UNIFORM, zero, dequantizeScale)
                    dequantizeLayer = network.add_scale(quantizeLayer.get_output(0), trt.ScaleMode.UNIFORM, zero, one)
                    dequantizeLayer.set_output_type(0, trt.float32)
                    dequantizeLayer.name = "InputDequantizeNode"
                    dequantizeLayer.get_output(0).name = "DequantizedInputTensor"
                    layer.set_input(k, dequantizeLayer.get_output(0))

engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
context.set_input_shape(lTensorName[0], [1, 1, nHeight, nWidth])
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
for i in range(nIO):
    bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
bufferD = []
for i in range(nIO):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

data = cv2.imread(inferenceImage, cv2.IMREAD_GRAYSCALE).astype(np.float32).reshape(1, 1, nHeight, nWidth)
bufferH[0] = data

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