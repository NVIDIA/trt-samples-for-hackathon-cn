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

soFilePath  = './OneHotPlugin.so'
np.random.seed(97)

def oneHotCPU(inputH0,shape,nEmbed,isFp16):
    output = np.zeros([np.prod(shape),nEmbed],dtype=[np.float32,np.float16][int(isFp16)])
    for i,x in enumerate(inputH0):
        output[i,x] = 1
    return output.reshape(shape+[nEmbed])

def getOneHotPlugin(nEmbed, isFp16):
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == 'OneHotPlugin':
            p0 = trt.PluginField("nEmbed", np.array([nEmbed],dtype=np.int32), trt.PluginFieldType.INT32)
            p1 = trt.PluginField("isFp16", np.array([isFp16],dtype=np.int32), trt.PluginFieldType.INT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([p0,p1]))
    return None

def buildEngine(logger, shape, nEmbed, isFp16):
    builder = trt.Builder(logger)
    builder.max_batch_size = 4
    network = builder.create_network()
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    config.flags = int(isFp16)

    inputT0 = network.add_input('inputT0', trt.int32, shape)
    oneHotLayer = network.add_plugin_v2([inputT0], getOneHotPlugin(nEmbed, int(isFp16)))

    network.mark_output(oneHotLayer.get_output(0))
    return builder.build_engine(network,config)

def run(nBatchSize, shape, nEmbed, isFp16):
    print("test", shape, nEmbed, 'Fp'+['32','16'][int(isFp16)])
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    engine = buildEngine(logger,shape,nEmbed,isFp16)
    if engine == None:
        print("Failed building engine!")
        return None
    print("Succeeded building engine!")

    context     = engine.create_execution_context()
    stream      = cuda.Stream()
    data        = np.random.randint(0,nEmbed,[nBatchSize]+shape).astype(np.int32)
    inputH0     = np.ascontiguousarray(data.reshape(-1))
    inputD0     = cuda.mem_alloc(inputH0.nbytes)
    outputH0    = np.empty((nBatchSize,)+tuple(context.get_binding_shape(1)), dtype=trt.nptype(engine.get_binding_dtype(1)))
    outputD0    = cuda.mem_alloc(outputH0.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    context.execute_async(nBatchSize, [int(inputD0), int(outputD0)], stream.handle)
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    stream.synchronize()

    outputH0CPU = oneHotCPU(inputH0,[nBatchSize]+shape,nEmbed,isFp16)

    print("Check result:",[ "True" if np.all( outputH0 == outputH0CPU ) else "False"][0])
    '''
    temp = outputH0
    print("hOut:", np.shape(temp), temp.dtype, np.mean(temp), np.var(temp), np.max(temp), np.min(temp))
    print(temp)
    temp = outputH0CPU
    print("hOutCPU:", np.shape(temp), temp.dtype, np.mean(temp), np.var(temp), np.max(temp), np.min(temp))
    print(temp)
    '''

if __name__ == '__main__':
    np.set_printoptions(precision = 3, linewidth = 100, suppress = True)
    run(4, [32], 8, False)    
    run(4, [16,8], 16, False)
    run(4, [4,4,4], 32, False)
    run(4, [4,4,4,4], 1024, False)
    run(4, [16,8], 2048, False) # large book

    run(4, [32], 8, True)
    run(4, [16,8], 16, True)
    run(4, [4,4,4], 32, True)
    run(4, [4,4,4,4], 1024, True)
    run(4, [16,8], 2048, True) # large book
    
    print("test finish!")

