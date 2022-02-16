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
import ctypes
import numpy as np
from datetime import datetime as dt
from cuda import cuda
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt

os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

np.random.seed(97)
tf.compat.v1.set_random_seed(97)
epsilon         = 1e-6
pbFile          = './model.pb'
onnxFile        = './model.onnx'
onnxSurgeonFile = './model-surgeon.onnx'
soFile          = './AddScalarPlugin.so'
trtFile         = './model.trt'

def check(a, b, weak = False, info=""): # 用于比较 TF 和 TRT 的输出结果
    if weak:
        res = np.all( np.abs(a - b) < epsilon )
    else:
        res = np.all( a == b )
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check %s:"%info,res,diff0,diff1)

def printArray(x,info="",n=5):          # 用于输出数组统计信息
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t',x.reshape(-1)[:n],x.reshape(-1)[-n:])
    #print('\t',x.reshape(-1)[:n])

nIn,cIn,hIn,wIn = 2,3,4,5
inputX  = np.random.rand(nIn,cIn,hIn,wIn).astype(np.float32).reshape([nIn,cIn,hIn,wIn])

os.system("rm -rf ./*.pb ./*.onnx ./*.trt ./*.o ./*.d ./*.so")
np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
cuda.cuInit(0)
cuda.cuDeviceGet(0)

# TensorFlow 中创建网络并保存为 .pb 文件 ------------------------------------------------------------
x       = tf.compat.v1.placeholder(tf.float32, [None,cIn,hIn,wIn], name='x')
_h1     = tf.multiply(x,1,name='node-0')    # 某些前处理
_h2     = tf.add(_h1,1,name='node-1')       # 想要替换的算子 / 模块
y       = tf.multiply(_h2,1,name='node-2')  # 某些后处理

tfConfig = tf.compat.v1.ConfigProto()
tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.compat.v1.Session(config=tfConfig)
sess.run(tf.compat.v1.global_variables_initializer())
outputTF = sess.run(y,feed_dict={x:inputX})

constantGraph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,['node-2'])
with tf.gfile.FastGFile("./model.pb", mode='wb') as f:
    f.write(constantGraph.SerializeToString())
sess.close()
print("Succeeded building model in TensorFlow!")

# 将 .pb 文件转换为 .onnx 文件 ----------------------------------------------------------------------
os.system("python -m tf2onnx.convert --input %s --output %s --inputs 'x:0' --outputs 'node-2:0' --opset 13"%(pbFile,onnxFile))
print("Succeeded converting model into onnx!")

# 将 .onnx 文件中 TensorRT 不原生支持的节点替换为 Plugin ---------------------------------------------
graph = gs.import_onnx(onnx.load(onnxFile))
graph.inputs[0].shape = ['bs',3,4,5]
graph.outputs[0].shape = ['bs',3,4,5]

for node in graph.nodes:
    if node.op == 'Add' and node.name == 'node-1':
        scalar = node.inputs[1].values
        pluginV = gs.Variable("MyAddPluginVariable-0", np.dtype(np.float32), None)
        pluginN = gs.Node("AddScalar","MyAddPluginNode-0",inputs=[node.inputs[0]],outputs=[pluginV], attrs={"scalar": float(scalar)})
        graph.nodes.append(pluginN)
        node.o().inputs[0] = pluginV
        node.outputs.clear()

graph.cleanup()
onnx.save(gs.export_onnx(graph), onnxSurgeonFile)
print("Succeeded inserting AddScalar node!")

# 编译 Plugin 为 .so 文件 ---------------------------------------------------------------------------
os.system("make")
print("Succeeded building AddScalar Plugin!")

# TensorRT 中加载 .onnx 和 .so 创建 engine ----------------------------------------------------------
logger  = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
ctypes.cdll.LoadLibrary(soFile)
builder = trt.Builder(logger)
network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config  = builder.create_builder_config()
config.max_workspace_size = 6 << 30
parser = trt.OnnxParser(network, logger)
with open(onnxFile, 'rb') as model:
    if not parser.parse( model.read() ):
        print ("Failed parsing ONNX file!")
        for error in range(parser.num_errors):
            print (parser.get_error(error))
        exit()
    print ("Succeeded parsing ONNX file!")

inputT0 = network.get_input(0)
inputT0.shape = [-1,cIn,hIn,wIn]
profile.set_shape(inputT0.name, [1,cIn,hIn,wIn], [2,cIn,hIn,wIn], [4,cIn,hIn,wIn])
config.add_optimization_profile(profile)
engineString    = builder.build_serialized_network(network,config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, 'wb') as f:
    f.write( engineString )
engine = trt.Runtime(logger).deserialize_cuda_engine( engineString )
context         = engine.create_execution_context()
context.set_binding_shape(0,[nIn,cIn,hIn,wIn])
_, stream       = cuda.cuStreamCreate(0)

inputH0     = np.ascontiguousarray(inputX.reshape(-1))
outputH0    = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
_,inputD0   = cuda.cuMemAllocAsync(inputH0.nbytes,stream)
_,outputD0  = cuda.cuMemAllocAsync(outputH0.nbytes,stream)

cuda.cuMemcpyHtoDAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, stream)
context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cuda.cuMemcpyDtoHAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, stream)
cuda.cuStreamSynchronize(stream)

printArray(inputX,"input")
printArray(outputTF,"TF")
printArray(outputH0,"TRT")
check(outputTF,outputH0,True)

cuda.cuStreamDestroy(stream)
cuda.cuMemFree(inputD0)
cuda.cuMemFree(outputD0)

print("Succeeded running model in TensorRT!")

