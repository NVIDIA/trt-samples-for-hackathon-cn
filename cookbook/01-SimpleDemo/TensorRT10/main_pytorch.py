#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
import sys
from collections import OrderedDict  # keep the order of the tensors implicitly

import tensorrt as trt
import torch

sys.path.append("/trtcookbook/include")
from utils import datatype_trt_to_torch

# yapf:disable

trt_file = "./model.trt"
input_tensor_name = "inputT0"
device = "cuda:0"
data = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)            # inference input data

def run():
    logger = trt.Logger(trt.Logger.ERROR)                                       # create Logger, available level: VERBOSE, INFO, WARNING, ERROR, INTERNAL_ERROR
    if os.path.isfile(trt_file):                                               # load engine from file and skip building process if it existed
        with open(trt_file, "rb") as f:
            engine_bytes = f.read()
        if engine_bytes == None:
            print("Fail getting serialized engine")
            return
        print("Succeed getting serialized engine")
    else:                                                                       # build a serialized network from scratch
        builder = trt.Builder(logger)                                           # create Builder
        config = builder.create_builder_config()                                # create BuidlerConfig to set attribution of the network
        network = builder.create_network()                                      # create Network
        profile = builder.create_optimization_profile()                         # create OptimizationProfile if using Dynamic-Shape mode
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)     # set workspace for the building process (all GPU memory is used by default)

        input_tensor = network.add_input(input_tensor_name, trt.float32, [-1, -1, -1])  # set input tensor of the network
        profile.set_shape(input_tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])  # set dynamic shape range of the input tensor
        config.add_optimization_profile(profile)                                # add the Optimization Profile into the BuilderConfig

        identity_layer = network.add_identity(input_tensor)                     # here is only an identity layer in this simple network, which the output is exactly equal to input
        network.mark_output(identity_layer.get_output(0))                       # mark the tensor for output

        engine_bytes = builder.build_serialized_network(network, config)        # create a serialized network from the network
        if engine_bytes == None:
            print("Fail building engine")
            return
        print("Succeed building engine")
        with open(trt_file, "wb") as f:                                        # save the serialized network as binaray file
            f.write(engine_bytes)
            print(f"Succeed saving engine ({trt_file})")

    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)          # create inference engine
    if engine == None:
        print("Fail getting engine for inference")
        return
    print("Succeed getting engine for inference")
    context = engine.create_execution_context()                                 # create Execution Context from the engine (analogy to a GPU context, or a CPU process)

    tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

    context.set_input_shape(input_tensor_name, data.shape)                      # set runtime size of input tensor if using Dynamic-Shape mode

    for name in tensor_name_list:                                               # Print information of input / output tensors
        mode = engine.get_tensor_mode(name)
        data_type = engine.get_tensor_dtype(name)
        buildtime_shape = engine.get_tensor_shape(name)
        runtime_shape = context.get_tensor_shape(name)
        print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {runtime_shape}, {name}")

    buffer = OrderedDict()                                                      # prepare the memory buffer on host and device
    for name in tensor_name_list:
        data_type = engine.get_tensor_dtype(name)
        runtime_shape = context.get_tensor_shape(name)
        buffer[name] = torch.empty(tuple(runtime_shape), dtype=datatype_trt_to_torch(data_type), device=device)

    buffer[input_tensor_name] = data.contiguous().to(device)                    # set runtime data, MUST use contiguous, it is a SERIOUS lesson

    for name in tensor_name_list:
        context.set_tensor_address(name, buffer[name].data_ptr())                # bind address of buffer to context

    context.execute_async_v3(0)                                                 # do inference computation

    for name in tensor_name_list:
        print(name)
        print(buffer[name])

if __name__ == "__main__":
    os.system("rm -rf ./*.trt")
    run()                                                                       # build a TensorRT engine and do inference
    run()                                                                       # load a TensorRT engine and do inference
