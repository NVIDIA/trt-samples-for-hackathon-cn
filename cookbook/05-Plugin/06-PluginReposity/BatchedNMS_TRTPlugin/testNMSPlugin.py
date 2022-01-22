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

import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from time import time

DATASIZE    = 3840
RETAINSIZE  = 2000
lockH       = 960
lockW       = 1024
dataFile    = 'batchedNMSIO.npz'

def getBatchedNMSPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == 'BatchedNMS_TRT':            
            p0 = trt.PluginField("shareLocation", np.array([1],dtype=np.int32), trt.PluginFieldType.INT32)
            p1 = trt.PluginField("backgroundLabelId", np.array([-1],dtype=np.int32), trt.PluginFieldType.INT32)
            p2 = trt.PluginField("numClasses", np.array([1],dtype=np.int32), trt.PluginFieldType.INT32)
            p3 = trt.PluginField("topK", np.array([DATASIZE],dtype=np.int32), trt.PluginFieldType.INT32)
            p4 = trt.PluginField("keepTopK", np.array([RETAINSIZE],dtype=np.int32), trt.PluginFieldType.INT32)
            p5 = trt.PluginField("scoreThreshold", np.array([0.7],dtype=np.float32), trt.PluginFieldType.FLOAT32)
            p6 = trt.PluginField("iouThreshold", np.array([0.7],dtype=np.float32), trt.PluginFieldType.FLOAT32)
            p7 = trt.PluginField("isNormalized", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([p0,p1,p2,p3,p4,p5,p6,p7]))
    return None

def buildEngine(logger):
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 6 << 30
    network = builder.create_network()
    
    tensor1 = network.add_input('data1', trt.DataType.FLOAT, (DATASIZE,1,4))
    tensor2 = network.add_input('data2', trt.DataType.FLOAT, (DATASIZE,1))
    scaleLayer = network.add_scale(tensor1, trt.ScaleMode.UNIFORM, np.array([0.0], dtype = np.float32), np.array([1/max(lockH,lockW)], dtype = np.float32), np.array([1.0], dtype = np.float32))
    nmsLayer = network.add_plugin_v2([scaleLayer.get_output(0), tensor2], getBatchedNMSPlugin())
    
    network.mark_output(nmsLayer.get_output(0))
    network.mark_output(nmsLayer.get_output(1))
    network.mark_output(nmsLayer.get_output(2))
    network.mark_output(nmsLayer.get_output(3))
    return builder.build_cuda_engine(network)
    
def run():
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    
    engine = buildEngine(logger)
    if engine == None:
        print("Failed building engine!")
        return None
    print("Succeeded building engine!")

    context = engine.create_execution_context()
    stream = cuda.Stream()

    data = np.load("./beforeNMSLarge.npz")['prop'][:DATASIZE]
    norm = max(lockH,lockW)
    data[:,:4] /= norm
    inputH0 = np.ascontiguousarray(data[:,:4].reshape((DATASIZE,1,4)))
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    inputH1 = np.ascontiguousarray(data[:,4].reshape((DATASIZE,1)))
    inputD1 = cuda.mem_alloc(inputH1.nbytes)
    outputH0 = np.empty(engine.get_binding_shape(2),dtype=np.int32)
    outputD0 = cuda.mem_alloc(outputH0.nbytes)
    outputH1 = np.empty(engine.get_binding_shape(3),dtype=np.float32)
    outputD1 = cuda.mem_alloc(outputH1.nbytes)
    outputH2 = np.empty(engine.get_binding_shape(4),dtype=np.float32)
    outputD2 = cuda.mem_alloc(outputH2.nbytes)
    outputH3 = np.empty(engine.get_binding_shape(5),dtype=np.float32)
    outputD3 = cuda.mem_alloc(outputH3.nbytes)
        
    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    cuda.memcpy_htod_async(inputD1, inputH1, stream)
    context.execute_async(1, [int(inputD0), int(inputD1), int(outputD0), int(outputD1),int(outputD2),int(outputD3)], stream.handle)
           
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    cuda.memcpy_dtoh_async(outputH1, outputD1, stream)
    cuda.memcpy_dtoh_async(outputH2, outputD2, stream)
    cuda.memcpy_dtoh_async(outputH3, outputD3, stream)
    stream.synchronize()
    
    print("input data:\n", data[:30,:])
    print(np.shape(outputH0), np.shape(outputH1), np.shape(outputH2), np.shape(outputH3))
    print("outputH0:\n", outputH0)
    print("outputH1:\n", (outputH1[:50,:] * norm))
    print("outputH2:\n", outputH2[:50])
    print("outputH3:\n", outputH3[:50])
    return None

if __name__ == '__main__':
    np.set_printoptions(precision = 3, linewidth = 200, suppress = True)
    run()

