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
import ctypes
import numpy as np
from cuda import cudart
from time import time
import tensorrt as trt

np.random.seed(97)

# HtoD-bound
nIn,cIn,hIn,wIn = 8,64,256,256
cOut,hW,wW      = 1,3,3

# Calculation-bound
#nIn,cIn,hIn,wIn = 8,64,128,128
#cOut,hW,wW      = 64,9,9

# DtoH-bound
#nIn,cIn,hIn,wIn = 8,64,128,128
#cOut,hW,wW      = 256,3,3

engineFile = "./engin.trt"

def getEngine():
    logger                      = trt.Logger(trt.Logger.ERROR)
    if os.path.isfile('./engine.trt'):                                          
        with open('./engine.trt', 'rb') as f:                                   
            engine = trt.Runtime(logger).deserialize_cuda_engine( f.read() )    
        if engine == None:                                                  
            print("Failed loading engine!")                                 
            return
        print("Succeeded loading engine!")
    else:                                                                       
        builder                     = trt.Builder(logger)
        network                     = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile                     = builder.create_optimization_profile()
        config                      = builder.create_builder_config()
        config.max_workspace_size   = 6<<30

        inputT0 = network.add_input('inputT0',trt.DataType.FLOAT,[-1,cIn,hIn,wIn])
        profile.set_shape(inputT0.name, (1,cIn,hIn,wIn),(nIn,cIn,hIn,wIn),(nIn*2,cIn,hIn,wIn))
        config.add_optimization_profile(profile)

        weight = np.random.rand(cOut,cIn,hW,wW).astype(np.float32)*2-1
        bias   = np.random.rand(cOut).astype(np.float32)*2-1
        _0 = network.add_convolution_nd(inputT0,cOut,[hW,wW],weight,bias)
        _0.padding_nd = (hW//2,wW//2)
        _1 = network.add_activation(_0.get_output(0),trt.ActivationType.RELU)

        network.mark_output(_1.get_output(0))
        engineString    = builder.build_serialized_network(network,config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(engineFile, 'wb') as f:
            f.write( engineString )
        engine = trt.Runtime(logger).deserialize_cuda_engine( engineString )        
    return engine

def run1(engine):
    context     = engine.create_execution_context()
    context.set_binding_shape(0,[nIn,cIn,hIn,wIn])
    _, stream   = cudart.cudaStreamCreate()

    data        = np.random.rand(nIn*cIn*hIn*wIn).astype(np.float32).reshape(nIn,cIn,hIn,wIn)
    inputH0     = np.ascontiguousarray(data.reshape(-1))
    outputH0    = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    _,inputD0   = cudart.cudaMallocAsync(inputH0.nbytes,stream)
    _,outputD0  = cudart.cudaMallocAsync(outputH0.nbytes,stream)

    # 完整一次推理
    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    # 数据拷贝 HtoD 计时
    for i in range(10):
        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        
    trtTimeStart = time()
    for i in range(30):
        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeEnd = time()
    print( "%6.3fms - 1 stream, DataCopyHtoD" %((trtTimeEnd - trtTimeStart)/30*1000) )
    
    # 推理计时
    for i in range(10):
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    
    trtTimeStart = time()
    for i in range(30):
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeEnd = time()
    print( "%6.3fms - 1 stream, Inference" %((trtTimeEnd - trtTimeStart)/30*1000) )

    # 数据拷贝 DtoH 计时
    for i in range(10):
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        
    trtTimeStart = time()
    for i in range(30):
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeEnd = time()
    print( "%6.3fms - 1 stream, DataCopyDtoH" %((trtTimeEnd - trtTimeStart)/30*1000) )
        
    # 总时间计时
    for i in range(10):
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    
    trtTimeStart = time()
    for i in range(30):
        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeEnd = time()
    print( "%6.3fms - 1 stream, DataCopy + Inference" %((trtTimeEnd - trtTimeStart)/30*1000) )

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)

def run2(engine):
    context = engine.create_execution_context()
    context.set_binding_shape(0,[nIn,cIn,hIn,wIn])
    _, stream0  = cudart.cudaStreamCreate()
    _, stream1  = cudart.cudaStreamCreate()
    _, event0   = cudart.cudaEventCreate()
    _, event1   = cudart.cudaEventCreate()
    
    data        = np.random.rand(nIn*cIn*hIn*wIn).astype(np.float32).reshape(nIn,cIn,hIn,wIn)
    inputSize   = trt.volume(context.get_binding_shape(0))*np.array([0],dtype=trt.nptype(engine.get_binding_dtype(0))).nbytes
    outputSize  = trt.volume(context.get_binding_shape(1))*np.array([0],dtype=trt.nptype(engine.get_binding_dtype(1))).nbytes    
    _,inputH0   = cudart.cudaHostAlloc(inputSize, cudart.cudaHostAllocWriteCombined)
    _,inputH1   = cudart.cudaHostAlloc(inputSize, cudart.cudaHostAllocWriteCombined)
    _,outputH0  = cudart.cudaHostAlloc(outputSize, cudart.cudaHostAllocWriteCombined)
    _,outputH1  = cudart.cudaHostAlloc(outputSize, cudart.cudaHostAllocWriteCombined)
    _,inputD0   = cudart.cudaMallocAsync(inputSize,stream0)
    _,inputD1   = cudart.cudaMallocAsync(inputSize,stream1)
    _,outputD0  = cudart.cudaMallocAsync(outputSize,stream0)
    _,outputD1  = cudart.cudaMallocAsync(outputSize,stream1)
    
    # 总时间计时
    for i in range(10):
        context.execute_async_v2([int(inputD0), int(outputD0)], stream0)

    trtTimeStart = time()
    cudart.cudaEventRecord(event1,stream1)
    
    for i in range(30):
        inputH,         outputH     = [inputH1,outputH1]    if i & 1 else [inputH0,outputH0]
        inputD,         outputD     = [inputD1,outputD1]    if i & 1 else [inputD0,outputD0]
        eventBefore,    eventAfter  = [event0,event1]       if i & 1 else [event1,event0]
        stream                      = stream1               if i & 1 else stream0

        cudart.cudaMemcpyAsync(inputD, inputH, inputSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        cudart.cudaStreamWaitEvent(stream,eventBefore,cudart.cudaEventWaitDefault)
        context.execute_async_v2([int(inputD), int(outputD)], stream)
        cudart.cudaEventRecord(eventAfter,stream)
        cudart.cudaMemcpyAsync(outputH, outputD, outputSize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    '''# 奇偶循环拆开写
    for i in range(30//2):
        cudart.cudaMemcpyAsync(inputD0, inputH0, inputSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream0)
        cudart.cudaStreamWaitEvent(stream0,event1,cudart.cudaEventWaitDefault)
        context.execute_async_v2([int(inputD0), int(outputD0)], stream0)
        cudart.cudaEventRecord(event0,stream0)
        cudart.cudaMemcpyAsync(outputH0, outputD0, outputSize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream0)

        cudart.cudaMemcpyAsync(inputD1, inputH1, inputSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream1)
        cudart.cudaStreamWaitEvent(stream1,event0,cudart.cudaEventWaitDefault)
        context.execute_async_v2([int(inputD1), int(outputD1)], stream1)
        cudart.cudaEventRecord(event1,stream1)
        cudart.cudaMemcpyAsync(outputH1, outputD1, outputSize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream1)
    '''
    cudart.cudaEventSynchronize(event1)
    trtTimeEnd = time()
    print( "%6.3fms - 2 stream, DataCopy + Inference" %((trtTimeEnd - trtTimeStart)/30*1000) )

if __name__ == '__main__':
    #os.system("rm -rf ./*.trt")
    cudart.cudaDeviceSynchronize()
    engine = getEngine()# 创建 engine
    run1(engine)        # 单 stream 推理
    run2(engine)        # 双 stream 推理

