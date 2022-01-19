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

onnxFileList = ["./model0.onnx","./model1.onnx"]
trtFileList  = ["./engine0.trt","./engine-plugin.trt","./engine1.trt"]
imageName    = "./8.png"

def buildEngine(logger, onnxFile):
    builder = trt.Builder(logger)
    builder.max_batch_size = 4
    builder.max_workspace_size = 3 << 30    
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    if not os.path.exists(onnxFile):
        print("failed finding onnx file!")
        return
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print ("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print (parser.get_error(error))
            return
    #network.get_input(0).shape = [1,1,28,28]
    #network.get_input(0).name = 'x'
    #network.get_output(0).name = 'y'
    return builder.build_cuda_engine(network)

def buildEnginePlugin(logger):
    builder = trt.Builder(logger)
    builder.max_batch_size = 4
    builder.max_workspace_size = 3 << 30    
    network = builder.create_network()

    inputTensor = network.add_input("pluginInput", trt.DataType.FLOAT, (1,7*7*64))    
    identityLayer = network.add_identity(inputTensor)    
    network.mark_output(identityLayer.get_output(0))    
    return builder.build_cuda_engine(network)

def run():
    logger = trt.Logger(trt.Logger.INFO)
    engineList = []

    if os.path.isfile(trtFileList[0]):                                                              # the first part of Onnx network
       with open(trtFileList[0], 'rb') as f:
           engineStr = f.read()
           engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
       if engine == None:
           print("Failed loading engine 0!")
           return
       print("succeeded loading engine 0!")
       engineList.append(engine)
    else:
        engine = buildEngine(logger, onnxFileList[0])
        if engine == None:
            print("Failed building engine 0!")
            return
        print("succeeded building engine 0!")
        engineList.append(engine)
        with open(trtFileList[0], 'wb') as f:
            f.write( engine.serialize() )

    if os.path.isfile(trtFileList[1]):                                                              # plugin added manually
       with open(trtFileList[1], 'rb') as f:
           engineStr = f.read()
           engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
       if engine == None:
           print("Failed loading engine 1!")
           return
       print("succeeded loading engine 1!")
       engineList.append(engine)
    else:
        engine = buildEnginePlugin(logger)
        if engine == None:
            print("Failed building engine 1!")
            return
        print("succeeded building engine 1!")
        engineList.append(engine)
        with open(trtFileList[1], 'wb') as f:
            f.write( engine.serialize() )
            
    if os.path.isfile(trtFileList[2]):                                                              # the second part of Onnx network
       with open(trtFileList[2], 'rb') as f:
           engineStr = f.read()
           engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
       if engine == None:
           print("Failed loading engine 2!")
           return
       print("succeeded loading engine 2!")
       engineList.append(engine)
    else:
        engine = buildEngine(logger, onnxFileList[1])
        if engine == None:
            print("Failed building engine 2!")
            return
        print("succeeded building engine 2!")
        engineList.append(engine)
        with open(trtFileList[2], 'wb') as f:
            f.write( engine.serialize() )
    
    stream  = cuda.Stream()
    contextList = [ engine.create_execution_context() for engine in engineList]
    
    data      = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    input1_h  = np.ascontiguousarray(data.reshape(-1))
    input1_d  = cuda.mem_alloc(input1_h.nbytes)
    output_d  = [ cuda.mem_alloc(trt.float32.itemsize * trt.volume(engineList[i].get_binding_shape(1))) for i in range(3)]
    output_h = np.empty(contextList[2].get_binding_shape(1),dtype = trt.nptype(engineList[2].get_binding_dtype(1)))
    
    cuda.memcpy_htod_async(input1_d, input1_h, stream)
    contextList[0].execute_async(1, [int(input1_d), int(output_d[0])], stream.handle)
    contextList[1].execute_async(1, [int(output_d[0]), int(output_d[1])], stream.handle)
    contextList[2].execute_async(1, [int(output_d[1]), int(output_d[2])], stream.handle)
    cuda.memcpy_dtoh_async(output_h, output_d[2], stream)
    stream.synchronize()
    
    print(output_h)
    print("test finish!")
    
if __name__ == '__main__':
    #[ os.remove(item) for item in glob("./*.trt") + glob("./*.cache")]
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run()
