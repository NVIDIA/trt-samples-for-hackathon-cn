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
from datetime import datetime as dt
from cuda import cuda
import tensorrt as trt
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

np.random.seed(97)
epsilon = 1e-6
tf.compat.v1.set_random_seed(97)
pbFile          = './model.pb'
onnxFile        = './model.onnx'
onnxSurgeonFile = './model-surgeon.onnx'
trtFile         = './model.trt'

def check(a, b, weak = False, info=""):
    if weak:
        res = np.all( np.abs(a - b) < epsilon )
    else:
        res = np.all( a == b )
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check %s:"%info,res,diff0,diff1)

def printArray(x,info="",n=5):
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t',x.reshape(-1)[:n],x.reshape(-1)[-n:])
    #print('\t',x.reshape(-1)[:n])

def test():
    nIn,cIn,hIn,wIn = 2,3,4,5
    inputX  = np.random.rand(nIn,cIn,hIn,wIn).astype(np.float32).reshape([nIn,cIn,hIn,wIn])

    # TensorFlow part ------------------------------------------------------------------------------
    x           = tf.compat.v1.placeholder(tf.float32, [None,cIn,hIn,wIn], name='x')
    layerNorm   = tf.keras.layers.LayerNormalization( \
                    axis=[1,2,3],
                    epsilon=epsilon,
                    center=True,
                    scale=True,
                    beta_initializer = tf.truncated_normal_initializer(mean=0,stddev=0.1),
                    gamma_initializer = tf.truncated_normal_initializer(mean=0,stddev=0.1),
                    beta_regularizer=None,
                    gamma_regularizer=None,
                    beta_constraint=None,
                    gamma_constraint=None,
                    trainable=False,
                    name='layerNorm'
                    )
    y           = layerNorm(x)

    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())
    outputTF = sess.run(y,feed_dict={x:inputX})

    printArray(outputTF,"TF")

    tfPara = {}                                                                                     # 保存权重 
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name,value = i.name,sess.run(i)
        print(name,value.shape)
        tfPara[name] = value
    np.savez("paraLayerNorm.npz",**tfPara)    
    #[ print(tensor.name) for tensor in tf.get_default_graph().as_graph_def().node ]
         
    constantGraph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,['layerNorm/batchnorm/add_1'])
    with tf.gfile.FastGFile("./model.pb", mode='wb') as f:
        f.write(constantGraph.SerializeToString())
    sess.close()

    # Convert to Onnx ------------------------------------------------------------------------------
    os.system("python -m tf2onnx.convert --input %s --output %s --inputs 'x:0' --outputs 'layerNorm/batchnorm/add_1:0'"%(pbFile,onnxFile))

    # Replace Layer Normalization Node in Onnx -----------------------------------------------------
    #os.system("make")
    '''    
    # Build Layer Normalization Plugin in TensorRT -------------------------------------------------


    # TensorRT part --------------------------------------------------------------------------------
    logger  = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config  = builder.create_builder_config()
    config.max_workspace_size = 6 << 30
    inputT0         = network.add_input('inputT0', trt.DataType.FLOAT, [-1,-1,-1,-1])
    profile.set_shape(inputT0.name, [1,1,1,1],[2,3,4,5],[4,6,8,10])
    config.add_optimization_profile(profile)

    pluginLayer     = network.add_plugin_v2([inputT0], getLayerNormPlugin(scalar))
    network.mark_output(pluginLayer.get_output(0))
    engineString    = builder.build_serialized_network(network,config)
    if engineString == None:
        print("Failed building engine!")
        return
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write( engineString )
    engine = trt.Runtime(logger).deserialize_cuda_engine( engineString )
    context         = engine.create_execution_context()
    context.set_binding_shape(0,[nIn,cIn,hIn])
    _, stream       = cuda.cuStreamCreate(0)

    inputH0     = np.ascontiguousarray(inputX.reshape(-1))
    outputH0    = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    _,inputD0   = cuda.cuMemAllocAsync(inputH0.nbytes,stream)
    _,outputD0  = cuda.cuMemAllocAsync(outputH0.nbytes,stream)

    cuda.cuMemcpyHtoDAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cuda.cuMemcpyDtoHAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, stream)
    cuda.cuStreamSynchronize(stream)

    printArray(outputH0,"TRT")

    cuda.cuStreamDestroy(stream)
    cuda.cuMemFree(inputD0)
    cuda.cuMemFree(outputD0)
    '''

if __name__ == '__main__':
    #os.system("rm -rf ./*.npz ./model.pb ./model.trt")
    cuda.cuInit(0)
    cuda.cuDeviceGet(0)
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)

    test()

    print("test finish!")
