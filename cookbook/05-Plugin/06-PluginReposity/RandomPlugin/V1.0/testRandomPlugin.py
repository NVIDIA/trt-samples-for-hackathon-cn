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

import ctypes
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

soFilePath  = './RandomPlugin.so'
np.random.seed(97)
cuRandSeed = 97

def getRandomPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == 'RandomPlugin':
            p0 = trt.PluginField("seed", np.array([cuRandSeed],dtype=np.int32), trt.PluginFieldType.INT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([p0]))
    return None

def buildEngine(logger, nRow, nCol):
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    
    inputT0 = network.add_input('inputT0', trt.float32, (nRow, nCol))
    randLayer = network.add_plugin_v2([inputT0], getRandomPlugin())
    
    network.mark_output(randLayer.get_output(0))
    network.mark_output(randLayer.get_output(1))
    return builder.build_cuda_engine(network)
    
def run(nRow, nCol):
    print("test: nRow=%d,nCol=%d"%(nRow,nCol))
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)
    engine = buildEngine(logger, nRow, nCol)
    if engine == None:
        print("Failed building engine!")
        return None
    print("Succeeded building engine!")

    context = engine.create_execution_context()
    stream = cuda.Stream()

    data = np.full((nRow,nCol),1,dtype=np.float32)                                                  # uniform distribution
    #data = np.tile(np.arange(0,nCol,1,dtype=np.float32),[nRow,1])                                   # non-uniform distribution    
    inputH0 = np.ascontiguousarray(data.reshape(-1))
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    outputH1 = np.empty(context.get_binding_shape(2), dtype=trt.nptype(engine.get_binding_dtype(2)))
    outputD0 = cuda.mem_alloc(outputH0.nbytes)
    outputD1 = cuda.mem_alloc(outputH1.nbytes)
        
    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    context.execute_async(1, [int(inputD0), int(outputD0), int(outputD1)], stream.handle)
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    cuda.memcpy_dtoh_async(outputH1, outputD1, stream)
    stream.synchronize()

    print("outputH0")
    print(np.shape(outputH0), "mean=%.2f,var=%.2f,max=%d,min=%d"%(np.mean(outputH0), np.var(outputH0), np.max(outputH0), np.min(outputH0)))
    print("outputH1")
    print(np.shape(outputH1), "mean=%.2f"%(np.mean(outputH1)))
    #print(outputH0)
    #print(outputH1)
 
if __name__ == '__main__':
    run(320,30)
    run(320,9)
    run(320,4)
    run(320,192)
    print("test finish!")

