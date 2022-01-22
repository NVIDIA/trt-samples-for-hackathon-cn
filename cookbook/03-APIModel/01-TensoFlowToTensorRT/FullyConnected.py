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
import cv2
import numpy as np
from datetime import datetime as dt
from cuda import cuda
import tensorrt as trt
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

np.random.seed(97)
epsilon = 1e-6
tf.compat.v1.set_random_seed(97)

def check(a, b, weak = False):
    if weak:
        res = np.all( np.abs(a - b) < epsilon )
    else:
        res = np.all( a == b )
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check:",res,diff0,diff1)

def printArray(x,info="",n=5):
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t',x.reshape(-1)[:n],x.reshape(-1)[-n:])
    #print('\t',x.reshape(-1)[:n])

def test_tf_nn_linalg_matmul():
    print("test tf.nn.linalg.matmul")
    nIn,cIn,hIn,wIn = 2,1,28,28
    cOut            = 32
    inputData = np.random.rand(nIn,hIn,wIn,cIn).astype(np.float32).reshape([nIn,hIn,wIn,cIn])                   # NHWC format
    
    # TensorFlow part ------------------------------------------------------------------------------
    x       = tf.compat.v1.placeholder(tf.float32, [None,hIn,wIn,cIn], name='x')
    weight  = tf.compat.v1.get_variable('w1', shape = [hIn*wIn*cIn,cOut], initializer = tf.truncated_normal_initializer(mean=0,stddev=0.1))
    _h1     = tf.reshape(x, [-1, hIn*wIn*cIn])
    y       = tf.linalg.matmul( \
                _h1,
                weight,
                transpose_a=False,
                transpose_b=False,
                adjoint_a=False,
                adjoint_b=False,
                a_is_sparse=False,
                b_is_sparse=False,
                name='y'
                )

    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())
    
    outputTF = sess.run(y,feed_dict={x:inputData})
    tfPara = {}                                                                                     # 保存权重 
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name,value = i.name,sess.run(i)
        print(name,value.shape)
        tfPara[name] = value
    np.savez("para_tf_nn_linalg_matmul.npz",**tfPara)
    sess.close()

    # TensorRT part --------------------------------------------------------------------------------
    logger  = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config  = builder.create_builder_config()
    inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, (-1,hIn,wIn,cIn))
    profile.set_shape(inputT0.name, (1,hIn,wIn,cIn),(nIn,hIn,wIn,cIn),(nIn*2,hIn,wIn,cIn))          # 范围覆盖住之后需要的值就好
    config.add_optimization_profile(profile)
    
    weight  = np.load('./para_tf_nn_linalg_matmul.npz')['w1:0'].transpose(1,0).reshape(-1)          # 读取权重
    _h1 = network.add_fully_connected(inputT0,cOut,weight,None)
    _h2 = network.add_shape(_h1.get_output(0))                                                      # 把最后两维的 (1,1) 去掉，对齐 TF 模型
    _h3 = network.add_slice(_h2.get_output(0),[0],[2],[1])
    _h4 = network.add_shuffle(_h1.get_output(0))
    _h4.set_input(1,_h3.get_output(0))
       
    network.mark_output(_h4.get_output(0))
    engineString    = builder.build_serialized_network(network,config)
    engine          = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context         = engine.create_execution_context()
    context.set_binding_shape(0,[nIn,hIn,wIn,cIn])
    _, stream       = cuda.cuStreamCreate(0)

    inputH0     = np.ascontiguousarray(inputData.reshape(-1))
    outputH0    = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    _,inputD0   = cuda.cuMemAllocAsync(inputH0.nbytes,stream)
    _,outputD0  = cuda.cuMemAllocAsync(outputH0.nbytes,stream)

    cuda.cuMemcpyHtoDAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cuda.cuMemcpyDtoHAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, stream)
    cuda.cuStreamSynchronize(stream)

    printArray(inputData,"input")
    #print(inputData)
    printArray(outputTF,"TF output")
    #print(outputTF)
    printArray(outputH0,"TRT output")
    #print(outputH0)
    check(outputTF,outputH0,True)

    cuda.cuStreamDestroy(stream)
    cuda.cuMemFree(inputD0)
    cuda.cuMemFree(outputD0)

def test_tf_layers_Dense():
    print("test test.tf.layers.Dense")
    nIn,cIn,hIn,wIn = 2,1,28,28
    cOut            = 32
    inputData = np.random.rand(nIn,hIn,wIn,cIn).astype(np.float32).reshape([nIn,hIn,wIn,cIn])                   # NHWC format
    
    # TensorFlow part ------------------------------------------------------------------------------
    x       = tf.compat.v1.placeholder(tf.float32, [None,hIn,wIn,cIn], name='x')
    _h1     = tf.reshape(x, [-1, hIn*wIn*cIn])
    fc      = tf.compat.v1.layers.Dense( \
                cOut,
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
                name='fullyConnected'
                )
    y       = fc(_h1)
    
    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())
    
    outputTF = sess.run(y,feed_dict={x:inputData})
    tfPara = {}                                                                                     # 保存权重 
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name,value = i.name,sess.run(i)
        print(name,value.shape)
        tfPara[name] = value
    np.savez("para_tf_layers_Dense.npz",**tfPara)
    sess.close()

    # TensorRT part --------------------------------------------------------------------------------
    logger  = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config  = builder.create_builder_config()
    inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, (-1,hIn,wIn,cIn))
    profile.set_shape(inputT0.name, (1,hIn,wIn,cIn),(nIn,hIn,wIn,cIn),(nIn*2,hIn,wIn,cIn))          # 范围覆盖住之后需要的值就好
    config.add_optimization_profile(profile)

    para    = np.load('./para_tf_layers_Dense.npz')  
    weight  = para['fullyConnected/kernel:0'].transpose(1,0).reshape(-1)
    bias    = para['fullyConnected/bias:0'].reshape(-1)
    _h1     = network.add_fully_connected(inputT0,cOut,weight,bias)
    _h2     = network.add_activation(_h1.get_output(0),trt.ActivationType.RELU)
    _h3     = network.add_shape(_h2.get_output(0))                                                  # 把最后两维的 (1,1) 去掉，对齐 TF 模型
    _h4     = network.add_slice(_h3.get_output(0),[0],[2],[1])
    _h5     = network.add_shuffle(_h2.get_output(0))
    _h5.set_input(1,_h4.get_output(0))
       
    network.mark_output(_h5.get_output(0))
    engineString    = builder.build_serialized_network(network,config)
    engine          = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context         = engine.create_execution_context()
    context.set_binding_shape(0,[nIn,hIn,wIn,cIn])
    _, stream       = cuda.cuStreamCreate(0)

    inputH0     = np.ascontiguousarray(inputData.reshape(-1))
    outputH0    = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    _,inputD0   = cuda.cuMemAllocAsync(inputH0.nbytes,stream)
    _,outputD0  = cuda.cuMemAllocAsync(outputH0.nbytes,stream)

    cuda.cuMemcpyHtoDAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cuda.cuMemcpyDtoHAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, stream)
    cuda.cuStreamSynchronize(stream)

    printArray(inputData,"input")
    #print(inputData)
    printArray(outputTF,"TF output")
    #print(outputTF)
    printArray(outputH0,"TRT output")
    #print(outputH0)
    check(outputTF,outputH0,True)

    cuda.cuStreamDestroy(stream)
    cuda.cuMemFree(inputD0)
    cuda.cuMemFree(outputD0)

def test_tf_keras_layers_Dense():
    print("test test.tf.keras.layers.Dense")
    nIn,cIn,hIn,wIn = 2,1,28,28
    cOut            = 32
    inputData = np.random.rand(nIn,hIn,wIn,cIn).astype(np.float32).reshape([nIn,hIn,wIn,cIn])                   # NHWC format
    
    # TensorFlow part ------------------------------------------------------------------------------
    x       = tf.compat.v1.placeholder(tf.float32, [None,hIn,wIn,cIn], name='x')
    _h1     = tf.reshape(x, [-1, hIn*wIn*cIn])
    fc      = tf.keras.layers.Dense( \
                cOut,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                bias_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                name='fullyConnected'
                )
    y       = fc(_h1)
    
    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())
    
    outputTF = sess.run(y,feed_dict={x:inputData})
    tfPara = {}                                                                                     # 保存权重 
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name,value = i.name,sess.run(i)
        print(name,value.shape)
        tfPara[name] = value
    np.savez("para_tf_keras_layers_Dense.npz",**tfPara)
    sess.close()

    # TensorRT part --------------------------------------------------------------------------------
    logger  = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config  = builder.create_builder_config()
    inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, (-1,hIn,wIn,cIn))
    profile.set_shape(inputT0.name, (1,hIn,wIn,cIn),(nIn,hIn,wIn,cIn),(nIn*2,hIn,wIn,cIn))          # 范围覆盖住之后需要的值就好
    config.add_optimization_profile(profile)

    para    = np.load('./para_tf_keras_layers_Dense.npz')  
    weight  = para['fullyConnected/kernel:0'].transpose(1,0).reshape(-1)
    bias    = para['fullyConnected/bias:0'].reshape(-1)
    _h1     = network.add_fully_connected(inputT0,cOut,weight,bias)
    _h2     = network.add_activation(_h1.get_output(0),trt.ActivationType.RELU)
    _h3     = network.add_shape(_h2.get_output(0))                                                  # 把最后两维的 (1,1) 去掉，对齐 TF 模型
    _h4     = network.add_slice(_h3.get_output(0),[0],[2],[1])
    _h5     = network.add_shuffle(_h2.get_output(0))
    _h5.set_input(1,_h4.get_output(0))
       
    network.mark_output(_h5.get_output(0))
    engineString    = builder.build_serialized_network(network,config)
    engine          = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context         = engine.create_execution_context()
    context.set_binding_shape(0,[nIn,hIn,wIn,cIn])
    _, stream       = cuda.cuStreamCreate(0)

    inputH0     = np.ascontiguousarray(inputData.reshape(-1))
    outputH0    = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    _,inputD0   = cuda.cuMemAllocAsync(inputH0.nbytes,stream)
    _,outputD0  = cuda.cuMemAllocAsync(outputH0.nbytes,stream)

    cuda.cuMemcpyHtoDAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cuda.cuMemcpyDtoHAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, stream)
    cuda.cuStreamSynchronize(stream)

    printArray(inputData,"input")
    #print(inputData)
    printArray(outputTF,"TF output")
    #print(outputTF)
    printArray(outputH0,"TRT output")
    #print(outputH0)
    check(outputTF,outputH0,True)

    cuda.cuStreamDestroy(stream)
    cuda.cuMemFree(inputD0)
    cuda.cuMemFree(outputD0)
    
if __name__ == '__main__':
    os.system("rm -rf ./*.npz ./model.pb ./model.trt")
    cuda.cuInit(0)
    cuda.cuDeviceGet(0)
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)

    test_tf_nn_linalg_matmul()
    test_tf_layers_Dense()
    test_tf_keras_layers_Dense()

    print("test finish!")

