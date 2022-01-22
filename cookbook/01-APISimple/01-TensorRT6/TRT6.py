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
import numpy as np
import tensorrt as trt
# cuda-python 仅支持 python>=3.7，更早版本只能使用 pycuda
import pycuda.autoinit
import pycuda.driver as cuda
#from cuda import cuda

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    if os.path.isfile('./engine.trt'):                                          
        with open('./engine.trt', 'rb') as f:                                   
            engine = trt.Runtime(logger).deserialize_cuda_engine( f.read() )    
        if engine == None:                                                  
            print("Failed loading engine!")                                 
            return
        print("Succeeded loading engine!")                                  
    else:                                                                       
        builder                     = trt.Builder(logger)
        builder.max_batch_size      = 3
        builder.max_workspace_size  = 1 << 30
        network                     = builder.create_network()

        inputTensor     = network.add_input('inputT0', trt.DataType.FLOAT, [4, 5])
        identityLayer   = network.add_identity(inputTensor)                                         # 恒等变换，什么都不做
        network.mark_output(identityLayer.get_output(0))
        engine          = builder.build_cuda_engine(network)
        if engine == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open('./engine.trt', 'wb') as f:
            f.write( engine.serialize() )

    context     = engine.create_execution_context()
    stream      = cuda.Stream()                                                                     # 使用 pycuda 的 API

    data        = np.arange(3*4*5,dtype=np.float32).reshape(3,4,5)
    inputH0     = np.ascontiguousarray(data.reshape(-1))
    outputH0    = np.empty((3,)+tuple(context.get_binding_shape(1)),dtype = trt.nptype(engine.get_binding_dtype(1)))
    inputD0     = cuda.mem_alloc(inputH0.nbytes)                                                    # 使用pycuda 的 API
    outputD0    = cuda.mem_alloc(outputH0.nbytes)                                                   # 使用pycuda 的 API

    cuda.memcpy_htod_async(inputD0,inputH0,stream)                                                  # 使用pycuda 的 API
    context.execute_async(3, [int(inputD0), int(outputD0)], stream.handle)                          # 使用pycuda 的 API
    cuda.memcpy_dtoh_async(outputH0,outputD0,stream)                                                # 使用pycuda 的 API
    stream.synchronize()                                                                            # 使用pycuda 的 API

    print("inputH0:", data.shape)
    print(data)
    print("outputH0:", outputH0.shape)
    print(outputH0)
    
    # 没有 cudaFree
    
if __name__ == '__main__':
    os.system("rm -rf ./*.trt")
    print( "GPU = %s"%(cuda.Device(0).name()) )    
    #cuda.Device(conf.iGPU).make_context()    
    run()                       # 创建 TensorRT 引擎并做推理
    run()                       # 读取 TensorRT 引擎并做推理
    #cuda.Context.pop()

