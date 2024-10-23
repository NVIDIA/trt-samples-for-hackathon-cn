#
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

import sys
from time import time

import numpy as np
import tensorrt as trt
from cuda import cudart

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1, case_mark

n_warm = 10
n_test = 30

@case_mark
def case_(nB, nC, nH, nW, nCOut, nHKernel, nWKernel):
    tw = TRTWrapperV1()

    inputTensor = tw.network.add_input("inputT0", trt.float32, [-1, nC, nH, nW])
    tw.profile.set_shape(inputTensor.name, [1, nC, nH, nW], [nB, nC, nH, nW], [nB * 2, nC, nH, nW])
    tw.config.add_optimization_profile(tw.profile)

    w = np.ascontiguousarray(np.random.rand(nCOut, nC, nHKernel, nWKernel).astype(np.float32) * 2 - 1)
    b = np.ascontiguousarray(np.random.rand(nCOut).astype(np.float32) * 2 - 1)
    layer = tw.network.add_convolution_nd(inputTensor, nCOut, [nHKernel, nWKernel], trt.Weights(w), trt.Weights(b))
    layer.padding_nd = (nHKernel // 2, nWKernel // 2)
    layer = tw.network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
    tensor = layer.get_output(0)
    tensor.name = "outputT0"
    tw.build([tensor])

    # Run with 1 CUDA stream
    _, stream = cudart.cudaStreamCreate()

    input_data = {"inputT0": np.ascontiguousarray(np.random.rand(nB * nC * nH * nW).astype(np.float32).reshape(nB, nC, nH, nW))}
    tw.setup(input_data)

    i = "inputT0"
    o = "outputT0"
    # Count time of memory copy from host to device
    for _ in range(n_warm):
        cudart.cudaMemcpyAsync(tw.buffer[i][1], tw.buffer[i][0].ctypes.data, tw.buffer[i][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaStreamSynchronize(stream)
    tik = time()
    for _ in range(n_test):
        cudart.cudaMemcpyAsync(tw.buffer[i][1], tw.buffer[i][0].ctypes.data, tw.buffer[i][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaStreamSynchronize(stream)
    tok = time()
    print(f"{(tok - tik) / n_test * 1000:6.3f}ms - 1 stream, DataCopyHtoD")

    # Count time of inference
    for _ in range(n_warm):
        tw.context.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)
    tik = time()
    for _ in range(n_test):
        tw.context.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)
    tok = time()
    print(f"{(tok - tik) / n_test * 1000:6.3f}ms - 1 stream, Inference")

    # Count time of memory copy from device to host
    for _ in range(n_warm):
        cudart.cudaMemcpyAsync(tw.buffer[o][0].ctypes.data, tw.buffer[o][1], tw.buffer[o][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    tik = time()
    for _ in range(n_test):
        cudart.cudaMemcpyAsync(tw.buffer[o][0].ctypes.data, tw.buffer[o][1], tw.buffer[o][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)
    tok = time()
    print(f"{(tok - tik) / n_test * 1000:6.3f}ms - 1 stream, DataCopyDtoH")

    # Count time of end-to-end
    for _ in range(n_warm):
        cudart.cudaMemcpyAsync(tw.buffer[i][1], tw.buffer[i][0].ctypes.data, tw.buffer[i][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        tw.context.execute_async_v3(stream)
        cudart.cudaMemcpyAsync(tw.buffer[o][0].ctypes.data, tw.buffer[o][1], tw.buffer[o][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)
    tik = time()
    for _ in range(n_test):
        cudart.cudaMemcpyAsync(tw.buffer[i][1], tw.buffer[i][0].ctypes.data, tw.buffer[i][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        tw.context.execute_async_v3(stream)
        cudart.cudaMemcpyAsync(tw.buffer[o][0].ctypes.data, tw.buffer[o][1], tw.buffer[o][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)
    tok = time()
    print(f"{(tok - tik) / n_test * 1000:6.3f}ms - 1 stream, DataCopy + Inference")

    # Run with 2 CUDA stream
    stream0 = cudart.cudaStreamCreate()[1]
    stream1 = cudart.cudaStreamCreate()[1]
    event0 = cudart.cudaEventCreate()[1]
    event1 = cudart.cudaEventCreate()[1]

    n_bytes_input = tw.buffer["inputT0"][2]
    n_bytes_output = tw.buffer["outputT0"][2]
    input_h0 = cudart.cudaHostAlloc(n_bytes_input, cudart.cudaHostAllocWriteCombined)[1]
    input_h1 = cudart.cudaHostAlloc(n_bytes_input, cudart.cudaHostAllocWriteCombined)[1]
    output_h0 = cudart.cudaHostAlloc(n_bytes_output, cudart.cudaHostAllocWriteCombined)[1]
    output_h1 = cudart.cudaHostAlloc(n_bytes_output, cudart.cudaHostAllocWriteCombined)[1]
    input_d0 = cudart.cudaMallocAsync(n_bytes_input, stream0)[1]
    input_d1 = cudart.cudaMallocAsync(n_bytes_input, stream1)[1]
    output_d0 = cudart.cudaMallocAsync(n_bytes_output, stream0)[1]
    output_d1 = cudart.cudaMallocAsync(n_bytes_output, stream1)[1]

    # Count time of end to end
    for _ in range(n_warm):
        tw.context.execute_async_v3(stream0)

    tik = time()
    cudart.cudaEventRecord(event1, stream1)

    for i in range(n_test):
        inputH, outputH = [input_h1, output_h1] if i & 1 else [input_h0, output_h0]
        inputD, outputD = [input_d1, output_d1] if i & 1 else [input_d0, output_d0]
        eventBefore, eventAfter = [event0, event1] if i & 1 else [event1, event0]
        stream = stream1 if i & 1 else stream0

        cudart.cudaMemcpyAsync(inputD, inputH, n_bytes_input, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        cudart.cudaStreamWaitEvent(stream, eventBefore, cudart.cudaEventWaitDefault)
        tw.context.execute_async_v3(stream)
        cudart.cudaEventRecord(eventAfter, stream)
        cudart.cudaMemcpyAsync(outputH, outputD, n_bytes_output, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    """
    # split the loop into odd and even iterations
    for i in range(n_test // 2):
        cudart.cudaMemcpyAsync(input_d0, input_h0, n_bytes_input, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream0)
        cudart.cudaStreamWaitEvent(stream0,event1,cudart.cudaEventWaitDefault)
        context.execute_async_v2([int(input_d0), int(output_d0)], stream0)
        cudart.cudaEventRecord(event0,stream0)
        cudart.cudaMemcpyAsync(output_h0, output_d0, n_bytes_output, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream0)

        cudart.cudaMemcpyAsync(input_d1, input_h1, n_bytes_input, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream1)
        cudart.cudaStreamWaitEvent(stream1,event0,cudart.cudaEventWaitDefault)
        context.execute_async_v2([int(input_d1), int(output_d1)], stream1)
        cudart.cudaEventRecord(event1,stream1)
        cudart.cudaMemcpyAsync(output_h1, output_d1, n_bytes_output, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream1)
    """
    cudart.cudaEventSynchronize(event1)
    tok = time()
    print(f"{(tok - tik) / n_test * 1000:6.3f}ms - 2 stream, DataCopy + Inference")

    for buffer in [input_h0, input_h1, output_h0, output_h1]:
        cudart.cudaFreeHost(buffer)
    for buffer in [input_d0, input_d1, output_d0, output_d1]:
        cudart.cudaFree(buffer)

if __name__ == "__main__":
    case_(8, 64, 256, 256, 1, 3, 3)  # HtoD bound
    case_(8, 64, 128, 128, 64, 9, 9)  # Compute bound
    case_(8, 64, 128, 128, 256, 3, 3)  # DtoH bound

    print("Finish")
