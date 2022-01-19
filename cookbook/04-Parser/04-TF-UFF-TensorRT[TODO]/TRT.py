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
import os
import cv2
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

uffFile = "./model.uff"
trtFile  = "./engine.trt"
imageName = "./8.png"

def buildEngine(logger):
    builder = trt.Builder(logger)
    builder.max_batch_size = 4
    builder.max_workspace_size = 3 << 30    
    network = builder.create_network()
    parser = trt.UffParser()

    if not os.path.exists(uffFile):
        print("failed finding uff file!")
        return
    parser.register_input("x", [28,28,1], trt.UffInputOrder.NHWC)
    parser.register_output("y")
    parser.parse(uffFile, network)
    
    return builder.build_cuda_engine(network)

def run():
    logger = trt.Logger(trt.Logger.INFO)
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engineStr = f.read()
            engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
        if engine == None:
            print("Failed loading engine!")
            return
        print("succeeded loading engine!")
    else:
        engine = buildEngine(logger)
        if engine == None:
            print("Failed building engine!")
            return
        print("succeeded building engine!")
        with open(trtFile, 'wb') as f:
            f.write( engine.serialize() )

    context = engine.create_execution_context()
    stream  = cuda.Stream()

    data      = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    input1_h  = np.ascontiguousarray(data.reshape(-1))
    input1_d  = cuda.mem_alloc(input1_h.nbytes)
    output1_h = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    output1_d = cuda.mem_alloc(output1_h.nbytes)
        
    cuda.memcpy_htod_async(input1_d, input1_h, stream)
    context.execute_async(1, [int(input1_d), int(output1_d)], stream.handle)
    cuda.memcpy_dtoh_async(output1_h, output1_d, stream)
    stream.synchronize()
    
    print(output1_h)
    print("test finish!")
    
if __name__ == '__main__':
    #[ os.remove(item) for item in glob("./*.trt") + glob("./*.cache")]
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run()
