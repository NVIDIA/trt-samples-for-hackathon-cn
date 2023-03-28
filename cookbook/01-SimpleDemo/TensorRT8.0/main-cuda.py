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

import os

import numpy as np
import tensorrt as trt
from cuda import cuda  # using CUDA  Driver API

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
    else:                                                                       # no .plan file, build engine from scratch
        builder = trt.Builder(logger)                                           # meta data of the network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30                                     # set workspace for TensorRT

        inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])   # set input tensor of the network
        profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])   # set dynamic shape range of the input tensor
        config.add_optimization_profile(profile)

        identityLayer = network.add_identity(inputTensor)                       # add a layer of identity operator
        network.mark_output(identityLayer.get_output(0))                        # set output tensor of the network

        engineString = builder.build_serialized_network(network, config)        # create serialized network from the networrk
        if engineString == None:
            print("Failed building serialized engine!")
            return
        print("Succeeded building serialized engine!")
        with open(trtFile, "wb") as f:                                          # save the serialized network as binaray file
            f.write(engineString)
            print("Succeeded saving .plan file!")

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)          # create TensorRT engine using Runtime
    if engine == None:
        print("Failed building engine!")
        return
    print("Succeeded building engine!")

    context = engine.create_execution_context()                                 # create CUDA context (similar to a process on GPU)
    context.set_input_shape(engine.get_tensor_name(0), [3, 4, 5])                                     # bind actual shape of the input tensor in Dynamic Shape mode
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])  # get information of the TensorRT engine
    nOutput = engine.num_bindings - nInput
    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput, nInput + nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    bufferH = []
    bufferH.append(np.ascontiguousarray(data))
    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cuda.cuMemAlloc(bufferH[i].nbytes)[1])

    for i in range(nInput):                                                     # copy the data from host to device
        cuda.cuMemcpyHtoD(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes)

    context.execute_v2(bufferD)                                                 # do inference computation

    for i in range(nInput, nInput + nOutput):                                   # copy the result from device to host
        cuda.cuMemcpyDtoH(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes)

    for i in range(nInput + nOutput):
        print(engine.get_binding_name(i))
        print(bufferH[i])

    for b in bufferD:                                                           # free the buffer on device
        cuda.cuMemFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    cuda.cuInit(0)                                                              # initialize the device manually
    cuda.cuDeviceGet(0)
    run()                                                                       # create TensorRT engine and do inference
    run()                                                                       # load TensorRT engine from file and do inference
