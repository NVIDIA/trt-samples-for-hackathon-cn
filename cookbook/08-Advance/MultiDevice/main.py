# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import OrderedDict  # keep the order of the tensors implicitly
from pathlib import Path

import numpy as np
import tensorrt as trt
import cuda.bindings.runtime as cudart

# yapf:disable

trt_file = Path("model.trt")
input_tensor_name = "inputT0"
data = np.arange(3 * 64, dtype=np.float32).reshape(3, 64)                  # Inference input data

def run():

    ############################################################################
    cudart.cudaSetDevice(0)  # Always use device 0 to build engine
    print(f"Current GPU = {cudart.cudaGetDevice()[1]}")
    ############################################################################

    logger = trt.Logger(trt.Logger.INFO)                                       # Create Logger, available level: VERBOSE, INFO, WARNING, ERROR, INTERNAL_ERROR
    if trt_file.exists():                                                       # Load engine from file and skip building process if it existed
        with open(trt_file, "rb") as f:
            engine_bytes = f.read()
        if engine_bytes == None:
            print("Fail getting serialized engine")
            return
        print("Succeed getting serialized engine")
    else:                                                                       # Build a serialized network from scratch
        builder = trt.Builder(logger)                                           # Create Builder
        config = builder.create_builder_config()                                # Create BuidlerConfig to set attribution of the network
        network = builder.create_network()                                      # Create Network
        profile = builder.create_optimization_profile()                         # Create OptimizationProfile if using Dynamic-Shape mode
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)     # Set workspace for the building process (all GPU memory is used by default)

        tensor = network.add_input(input_tensor_name, trt.float32, [-1, 64])  # Set input tensor of the network
        profile.set_shape(tensor.name, [1, 64], [3, 64], [6, 64])  # Set dynamic shape range of the input tensor
        config.add_optimization_profile(profile)                                # Add the Optimization Profile into the BuilderConfig

        # We build a "complex" network to see the performance differences
        for i in range(2048):
            w = np.random.rand(64, 64).astype(np.float32)
            b = np.random.rand(1, 64).astype(np.float32)
            layer_weight = network.add_constant(w.shape, trt.Weights(np.ascontiguousarray(w)))
            layer = network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.NONE)
            layer_bias = network.add_constant(b.shape, trt.Weights(np.ascontiguousarray(b)))
            layer = network.add_elementwise(layer.get_output(0), layer_bias.get_output(0), trt.ElementWiseOperation.SUM)
            layer = network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
            tensor = layer.get_output(0)

        network.mark_output(tensor)                       # Mark the tensor for output

        engine_bytes = builder.build_serialized_network(network, config)        # Create a serialized network from the network
        if engine_bytes == None:
            print("Fail building engine")
            return
        print("Succeed building engine")
        with open(trt_file, "wb") as f:                                         # Save the serialized network as binaray file
            f.write(engine_bytes)
            print(f"Succeed saving engine ({trt_file})")

    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)          # Create inference engine
    if engine == None:
        print("Fail getting engine for inference")
        return
    print("Succeed getting engine for inference")

    ############################################################################
    cudart.cudaSetDevice(0)  # Use device 0 to run
    print(f"Current GPU = {cudart.cudaGetDevice()[1]}")
    ############################################################################

    print("context create on GPU 0")
    context = engine.create_execution_context()                                 # Create Execution Context from the engine (analogy to a GPU context, or a CPU process)

    tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

    context.set_input_shape(input_tensor_name, data.shape)                      # Set runtime size of input tensor if using Dynamic-Shape mode

    for name in tensor_name_list:                                               # Print information of input / output tensors
        mode = engine.get_tensor_mode(name)
        data_type = engine.get_tensor_dtype(name)
        buildtime_shape = engine.get_tensor_shape(name)
        runtime_shape = context.get_tensor_shape(name)
        print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {runtime_shape}, {name}")

    buffer = OrderedDict()                                                      # Prepare the memory buffer on host and device
    for name in tensor_name_list:
        data_type = engine.get_tensor_dtype(name)
        runtime_shape = context.get_tensor_shape(name)
        n_byte = trt.volume(runtime_shape) * np.dtype(trt.nptype(data_type)).itemsize
        host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
        device_buffer = cudart.cudaMalloc(n_byte)[1]
        buffer[name] = [host_buffer, device_buffer, n_byte]

    buffer[input_tensor_name][0] = np.ascontiguousarray(data)                   # Set runtime data, MUST use np.ascontiguousarray, it is a SERIOUS lesson

    for name in tensor_name_list:
        context.set_tensor_address(name, buffer[name][1])                       # Bind address of device buffer to context

    # Do inference once before CUDA graph capture update internal state
    context.execute_async_v3(0)

    # CUDA Graph capture
    print("CUDA graph capture on GPU 0")
    _, stream = cudart.cudaStreamCreate()
    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)

    for name in tensor_name_list:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cudart.cudaMemcpyAsync(buffer[name][1], buffer[name][0].ctypes.data, buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    context.execute_async_v3(stream)

    for name in tensor_name_list:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            cudart.cudaMemcpyAsync(buffer[name][0].ctypes.data, buffer[name][1], buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    #cudart.cudaStreamSynchronize(stream)  # Do not synchronize during capture
    _, graph = cudart.cudaStreamEndCapture(stream)

    """
    # CUDA graph launch
    print("CUDA graph launch on GPU 0")
    _, graphExe = cudart.cudaGraphInstantiate(graph, 0)
    cudart.cudaGraphLaunch(graphExe, stream)
    cudart.cudaStreamSynchronize(stream)

    for name in tensor_name_list:
        print(name)
        print(buffer[name][0])
    """
    for _, device_buffer, _ in buffer.values():                                 # Free the GPU memory buffer after all work
        cudart.cudaFree(device_buffer)

    ############################################################################
    cudart.cudaSetDevice(1)  # Use device 1 to run
    print(f"Current GPU = {cudart.cudaGetDevice()[1]}")
    ############################################################################

    print("context create on GPU 1")
    context = engine.create_execution_context()                                 # Create Execution Context from the engine (analogy to a GPU context, or a CPU process)

    tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

    context.set_input_shape(input_tensor_name, data.shape)                      # Set runtime size of input tensor if using Dynamic-Shape mode

    for name in tensor_name_list:                                               # Print information of input / output tensors
        mode = engine.get_tensor_mode(name)
        data_type = engine.get_tensor_dtype(name)
        buildtime_shape = engine.get_tensor_shape(name)
        runtime_shape = context.get_tensor_shape(name)
        print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {runtime_shape}, {name}")

    buffer = OrderedDict()                                                      # Prepare the memory buffer on host and device
    for name in tensor_name_list:
        data_type = engine.get_tensor_dtype(name)
        runtime_shape = context.get_tensor_shape(name)
        n_byte = trt.volume(runtime_shape) * np.dtype(trt.nptype(data_type)).itemsize
        host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
        device_buffer = cudart.cudaMalloc(n_byte)[1]
        buffer[name] = [host_buffer, device_buffer, n_byte]

    buffer[input_tensor_name][0] = np.ascontiguousarray(data)                   # Set runtime data, MUST use np.ascontiguousarray, it is a SERIOUS lesson

    for name in tensor_name_list:
        context.set_tensor_address(name, buffer[name][1])                       # Bind address of device buffer to context

    # Do inference once before CUDA graph capture update internal state
    context.execute_async_v3(0)

    # CUDA Graph capture
    print("CUDA graph capture on GPU 1")
    _, stream = cudart.cudaStreamCreate()
    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)

    for name in tensor_name_list:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cudart.cudaMemcpyAsync(buffer[name][1], buffer[name][0].ctypes.data, buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    context.execute_async_v3(stream)

    for name in tensor_name_list:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            cudart.cudaMemcpyAsync(buffer[name][0].ctypes.data, buffer[name][1], buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    #cudart.cudaStreamSynchronize(stream)  # Do not synchronize during capture
    _, graph = cudart.cudaStreamEndCapture(stream)

    """
    # CUDA graph launch
    print("CUDA graph launch on GPU 1")
    _, graphExe = cudart.cudaGraphInstantiate(graph, 0)
    cudart.cudaGraphLaunch(graphExe, stream)
    cudart.cudaStreamSynchronize(stream)

    for name in tensor_name_list:
        print(name)
        print(buffer[name][0])
    """
    for _, device_buffer, _ in buffer.values():                                 # Free the GPU memory buffer after all work
        cudart.cudaFree(device_buffer)



if __name__ == "__main__":
    os.system("rm -rf *.trt")

    run()                                                                       # Build a TensorRT engine and do inference
    run()                                                                       # Load a TensorRT engine and do inference

    print("Finish")
