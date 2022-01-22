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
from cuda import cuda

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
    _, stream   = cuda.cuStreamCreate(0)

    data        = np.arange(3*4*5,dtype=np.float32).reshape(3,4,5)
    inputH0     = np.ascontiguousarray(data.reshape(-1))
    outputH0    = np.empty((3,)+tuple(context.get_binding_shape(1)),dtype = trt.nptype(engine.get_binding_dtype(1)))
    _,inputD0   = cuda.cuMemAllocAsync(inputH0.nbytes,stream)
    _,outputD0  = cuda.cuMemAllocAsync(outputH0.nbytes,stream)

    cuda.cuMemcpyHtoDAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, stream)
    context.execute_async(3, [int(inputD0), int(outputD0)], stream)
    cuda.cuMemcpyDtoHAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, stream)
    cuda.cuStreamSynchronize(stream)

    print("inputH0:", data.shape)
    print(data)
    print("outputH0:", outputH0.shape)
    print(outputH0)

    cuda.cuStreamDestroy(stream)
    cuda.cuMemFree(inputD0)
    cuda.cuMemFree(outputD0)

if __name__ == '__main__':
    os.system("rm -rf ./*.trt")
    cuda.cuInit(0)
    cuda.cuDeviceGet(0)
    run()                       # 创建 TensorRT 引擎并做推理
    run()                       # 读取 TensorRT 引擎并做推理

