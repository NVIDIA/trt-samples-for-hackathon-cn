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

import os
import sys
import numpy as np
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt

os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
np.random.seed(31193)
tf.compat.v1.set_random_seed(97)
nB, nC, nH, nW = 2, 1, 28, 28
cOut, hW, wW = 32, 5, 5
inputData = np.random.rand(nB, nH, nW, nC).astype(np.float32).reshape([nB, nH, nW, nC])  # NHWC format

def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))

def printArrayInfomation(x, info="", n=5):
    print( "%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f"%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print("\t", x.reshape(-1)[:n], x.reshape(-1)[-n:])
    #print("\t",x.reshape(-1)[:n])

def test_tf_nn_conv2d():
    print("\ntf.nn.conv2d ------------------------------------------------------")
    # TensorFlow part ----------------------------------------------------------
    x = tf.compat.v1.placeholder(tf.float32, [None, nH, nW, nC], name="x")
    weight = tf.compat.v1.get_variable("w1", shape=[hW, wW, nC, cOut], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    y       = tf.nn.conv2d( \
                x,
                filter=weight,
                strides=None,
                padding="SAME",
                use_cudnn_on_gpu=True,
                data_format="NHWC",
                dilations=[1, 1, 1, 1],
                name="y",
                filters=None
                )

    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())

    outputTF = sess.run(y, feed_dict={x: inputData})
    tfPara = {}  # save weight as file
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name, value = i.name, sess.run(i)
        print(name, value.shape)
        tfPara[name] = value
    np.savez("para_tf_nn_conv2d.npz", **tfPara)
    sess.close()

    # TensorRT part ------------------------------------------------------------
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    inputT0 = network.add_input("inputT0", trt.float32, (-1, -1, -1, nC))
    profile.set_shape(inputT0.name, [1, 1, 1, nC], [nB, nH, nW, nC], [nB * 2, nH * 2, nW * 2, nC])
    config.add_optimization_profile(profile)

    _h1 = network.add_shuffle(inputT0)  # NHWC to NCHW
    _h1.first_transpose = (0, 3, 1, 2)
    weight = np.load("./para_tf_nn_conv2d.npz")["w1:0"].transpose(3, 2, 0, 1).reshape(-1)
    _h2 = network.add_convolution_nd(_h1.get_output(0), cOut, [hW, wW], weight, None)
    _h2.padding_nd = (2, 2)
    _h3 = network.add_shuffle(_h2.get_output(0))  # NCHW to NHWC, align with TF
    _h3.first_transpose = (0, 2, 3, 1)

    network.mark_output(_h3.get_output(0))
    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()
    context.set_input_shape(engine.get_tensor_name(0), [nB, nH, nW, nC])
    _, stream = cudart.cudaStreamCreate()

    inputH0 = np.ascontiguousarray(inputData.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    printArrayInfomation(inputData, "input")
    #print(inputData)
    printArrayInfomation(outputTF, "TF output")
    #print(outputTF)
    printArrayInfomation(outputH0, "TRT output")
    #print(outputH0)
    check(outputTF, outputH0, True)

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)

def test_tf_layers_Conv2D():
    print("\ntf.layers.Conv2D --------------------------------------------------")
    # TensorFlow part ----------------------------------------------------------
    x = tf.compat.v1.placeholder(tf.float32, [None, nH, nW, nC], name="x")
    conv    = tf.compat.v1.layers.Conv2D( \
                cOut,
                [hW,wW],
                strides=(1, 1),
                padding="same",
                data_format="channels_last",
                dilation_rate=(1, 1),
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                bias_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=False,
                name="convolution",
                )
    y = conv(x)

    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())

    outputTF = sess.run(y, feed_dict={x: inputData})
    tfPara = {}  # save weight as file
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name, value = i.name, sess.run(i)
        print(name, value.shape)
        tfPara[name] = value
    np.savez("para_tf_layers_Conv2D.npz", **tfPara)
    sess.close()

    # TensorRT part ------------------------------------------------------------
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    inputT0 = network.add_input("inputT0", trt.float32, (-1, -1, -1, nC))
    profile.set_shape(inputT0.name, [1, 1, 1, nC], [nB, nH, nW, nC], [nB * 2, nH * 2, nW * 2, nC])
    config.add_optimization_profile(profile)

    _h1 = network.add_shuffle(inputT0)  # NHWC to NCHW
    _h1.first_transpose = (0, 3, 1, 2)
    para = np.load("./para_tf_layers_Conv2D.npz")
    weight = np.ascontiguousarray(para["convolution/kernel:0"].transpose(3, 2, 0, 1).reshape(-1))
    bias = np.ascontiguousarray(para["convolution/bias:0"].reshape(-1))
    _h2 = network.add_convolution_nd(_h1.get_output(0), cOut, [hW, wW], weight, bias)
    _h2.padding_nd = (2, 2)
    _h3 = network.add_activation(_h2.get_output(0), trt.ActivationType.RELU)
    _h4 = network.add_shuffle(_h3.get_output(0))  # NCHW to NHWC, align with TF
    _h4.first_transpose = (0, 2, 3, 1)

    network.mark_output(_h4.get_output(0))
    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()
    context.set_input_shape(engine.get_tensor_name(0), [nB, nH, nW, nC])
    _, stream = cudart.cudaStreamCreate()

    inputH0 = np.ascontiguousarray(inputData.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    printArrayInfomation(inputData, "input")
    #print(inputData)
    printArrayInfomation(outputTF, "TF output")
    #print(outputTF)
    printArrayInfomation(outputH0, "TRT output")
    #print(outputH0)
    check(outputTF, outputH0, True)

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)

def test_tf_keras_layer_Conv2D():
    print("\ntf.keras.layers.Conv2D --------------------------------------------")
    # TensorFlow part ----------------------------------------------------------
    x = tf.compat.v1.placeholder(tf.float32, [None, nH, nW, nC], name="x")
    conv    = tf.keras.layers.Conv2D( \
                cOut,
                [hW,wW],
                strides=(1, 1),
                padding="same",
                data_format="channels_last",
                dilation_rate=(1, 1),
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                bias_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None
                )
    y = conv(x)

    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())

    outputTF = sess.run(y, feed_dict={x: inputData})
    tfPara = {}  # save weight as file
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name, value = i.name, sess.run(i)
        print(name, value.shape)
        tfPara[name] = value
    np.savez("para_tf_keras_layer_Conv2D.npz", **tfPara)
    sess.close()

    # TensorRT part ------------------------------------------------------------
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    inputT0 = network.add_input("inputT0", trt.float32, (-1, -1, -1, nC))
    profile.set_shape(inputT0.name, [1, 1, 1, nC], [nB, nH, nW, nC], [nB * 2, nH * 2, nW * 2, nC])
    config.add_optimization_profile(profile)

    _h1 = network.add_shuffle(inputT0)  # NHWC to NCHW
    _h1.first_transpose = (0, 3, 1, 2)
    para = np.load("./para_tf_keras_layer_Conv2D.npz")
    weight = np.ascontiguousarray(para["conv2d/kernel:0"].transpose(3, 2, 0, 1).reshape(-1))
    bias = np.ascontiguousarray(para["conv2d/bias:0"].reshape(-1))
    _h2 = network.add_convolution_nd(_h1.get_output(0), cOut, [hW, wW], weight, bias)
    _h2.padding_nd = (2, 2)
    _h3 = network.add_activation(_h2.get_output(0), trt.ActivationType.RELU)
    _h4 = network.add_shuffle(_h3.get_output(0))  # NCHW to NHWC, align with TF
    _h4.first_transpose = (0, 2, 3, 1)

    network.mark_output(_h4.get_output(0))
    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()
    context.set_input_shape(engine.get_tensor_name(0), [nB, nH, nW, nC])
    _, stream = cudart.cudaStreamCreate()

    inputH0 = np.ascontiguousarray(inputData.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    printArrayInfomation(inputData, "input")
    #print(inputData)
    printArrayInfomation(outputTF, "TF output")
    #print(outputTF)
    printArrayInfomation(outputH0, "TRT output")
    #print(outputH0)
    check(outputTF, outputH0, True)

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)

if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    np.set_printoptions(precision=3, linewidth=100, suppress=True)

    test_tf_nn_conv2d()
    test_tf_layers_Conv2D()
    test_tf_keras_layer_Conv2D()

    print("\ntest finish!")
