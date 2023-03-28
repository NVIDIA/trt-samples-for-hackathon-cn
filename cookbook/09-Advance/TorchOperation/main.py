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

import tensorrt as trt
import torch

trtFile = "./model.plan"

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()

        inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
        profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
        config.add_optimization_profile(profile)

        identityLayer = network.add_identity(inputTensor)
        network.mark_output(identityLayer.get_output(0))

        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building serialized engine!")
            return
        print("Succeeded building serialized engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
            print("Succeeded saving .plan file!")

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    if engine == None:
        print("Failed building engine!")
        return
    print("Succeeded building engine!")

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    stream = torch.cuda.Stream()

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], [3, 4, 5])
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    buffer = []  # do not distinguish buffer between host and device
    buffer.append(torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5))
    for i in range(nInput, nIO):
        if engine.get_tensor_dtype(lTensorName[i]) == trt.float32:
            dataType = torch.float32
        else:
            print("Not implement")
            dataType = torch.float32
        buffer.append(torch.empty(tuple(context.get_tensor_shape(lTensorName[i])), dtype=dataType).cuda())

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(buffer[i].data_ptr()))

    context.execute_async_v3(stream.cuda_stream)

    for i in range(nIO):
        print(lTensorName[i])
        print(buffer[i])

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    run()
    run()
