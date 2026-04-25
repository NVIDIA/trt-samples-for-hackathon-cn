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
#

import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import TRTWrapperV1, build_mnist_network_trt, case_mark

trt_file = Path("model.trt")
data_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data" / "InferenceData.npy"
input_data = {"x": np.load(data_file)}

@case_mark
def case_build():
    import tensorrt as trt
    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)  # Set the flag of version compatibility

    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)
    tw.build(output_tensor_list)
    tw.serialize_engine(trt_file)

@case_mark
def case_normal():
    import tensorrt as trt
    tw = TRTWrapperV1(logger="VERBOSE", trt_file=trt_file)  # Use VERBOSE log to see resource consumption
    tw.runtime = trt.Runtime(tw.logger)  # We need to initialize a runtime outside tw since we must enable a switch here
    tw.runtime.engine_host_code_allowed = True  # Turn on the switch
    tw.setup(input_data)
    tw.infer()

# Rewrite runtime since we can not use `tw.setup(data)` or `tw.infer()` directly
def runtime_for_lean_or_dispatch(trt):

    with open(trt_file, "rb") as f:
        engine_bytes = f.read()

    logger = trt.Logger(trt.Logger.VERBOSE)  # Use VERBOSE log to see resource consumption
    runtime = trt.Runtime(logger)
    runtime.engine_host_code_allowed = True
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    n_input = sum([engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in tensor_name_list])
    engine.num_io_tensors - n_input

    context = engine.create_execution_context()
    for name, data in input_data.items():
        context.set_input_shape(name, data.shape)

    buffer = OrderedDict()
    for name in tensor_name_list:
        data_type = engine.get_tensor_dtype(name)
        runtime_shape = context.get_tensor_shape(name)
        n_byte = trt.volume(runtime_shape) * data_type.itemsize
        host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
        device_buffer = cudart.cudaMalloc(n_byte)[1]
        buffer[name] = [host_buffer, device_buffer, n_byte]

    for name, data in input_data.items():
        buffer[name][0] = np.ascontiguousarray(data)

    for name in tensor_name_list:
        context.set_tensor_address(name, buffer[name][1])

    for name in tensor_name_list:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cudart.cudaMemcpy(buffer[name][1], buffer[name][0].ctypes.data, buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_async_v3(0)

    for name in tensor_name_list:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            cudart.cudaMemcpy(buffer[name][0].ctypes.data, buffer[name][1], buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    if True:
        for name in tensor_name_list:
            print(name)
            print(buffer[name][0])

    for _, device_buffer, _ in buffer.values():
        cudart.cudaFree(device_buffer)
    return

@case_mark
def case_lean():
    import tensorrt_lean as trtl
    runtime_for_lean_or_dispatch(trtl)

@case_mark
def case_dispatch():
    import tensorrt_dispatch as trtd
    runtime_for_lean_or_dispatch(trtd)

if __name__ == "__main__":
    for trt_path in Path(".").glob("*.trt"):
        trt_path.unlink(missing_ok=True)

    case_build()
    case_normal()
    case_lean()
    case_dispatch()

    print("Finish")
