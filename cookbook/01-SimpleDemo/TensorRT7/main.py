#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

from cuda import cudart  # using CUDA Runtime API
import numpy as np
import os
import tensorrt as trt

# yapf:disable

trtFile = "./model.plan"
data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)                  # input data for inference

def run():
    logger = trt.Logger(trt.Logger.ERROR)                                       # Logger, avialable level: VERBOSE, INFO, WARNING, ERRROR, INTERNAL_ERROR
    if os.path.isfile(trtFile):                                                 # read .plan file if exists
        with open(trtFile, "rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)      # deserialize the binaray object into TensorRT engine
        if engine == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
    else:                                                                       # no .plan file, build engine from scratch
        builder = trt.Builder(logger)                                           # meta data of the network
        builder.max_batch_size = 3
        builder.max_workspace_size = 1 << 30                                    # set workspace for TensorRT
        network = builder.create_network()

        inputTensor = network.add_input("inputT0", trt.float32, [4, 5])         # set input tensor of the network

        identityLayer = network.add_identity(inputTensor)                       # add a layer of identity operator
        network.mark_output(identityLayer.get_output(0))                        # set output tensor of the network

        engine = builder.build_cuda_engine(network)                             # create TensorRT engine from the networrk
        if engine == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:                                          # serialize the TensorRT engine as binaray file
            f.write(engine.serialize())
            print("Succeeded saving .plan file!")

    context = engine.create_execution_context()                                 # create CUDA context (similar to a process on GPU)
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])  # get information of the TensorRT engine
    nOutput = engine.num_bindings - nInput
    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput, nInput + nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    bufferH = []
    bufferH.append(np.ascontiguousarray(data))
    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty((3, ) + tuple(context.get_binding_shape(i)), dtype=trt.nptype(engine.get_binding_dtype(i))))
    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):                                                     # copy the data from host to device
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute(3, bufferD)                                                 # do inference computation

    for i in range(nInput, nInput + nOutput):                                   # copy the result from device to host
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nInput + nOutput):
        print(engine.get_binding_name(i))
        print(bufferH[i].reshape((3, ) + tuple(context.get_binding_shape(i))))

    for b in bufferD:                                                           # free the buffer on device
        cudart.cudaFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    run()                                                                       # create TensorRT engine and do inference
    run()                                                                       # load TensorRT engine from file and do inference
