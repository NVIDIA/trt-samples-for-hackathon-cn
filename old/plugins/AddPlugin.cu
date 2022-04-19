/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "AddPlugin.h"
#include "cuda_fp16.h"
#include <chrono>
#include <thread>

template<typename T>
__global__ void AddValue(T *pDst, T *pSrc, int n, T valueToAdd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    pDst[x] = pSrc[x] + valueToAdd;
}

int AddPlugin::enqueue(int nBatch, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    int n = nBatch;
    for (int i = 0; i < m.inputDim.nbDims; i++) {
        n *= m.inputDim.d[i];
    }
    printf("n=%d, nBatch=%d\n", n, nBatch);
    if (m.dataType == nvinfer1::DataType::kFLOAT) {
        std::cout << "Running fp32 kernel" << std::endl;
        std::this_thread::sleep_for(20ms);
        AddValue<<<(n + 1023) / 1024, 1024>>>((float *)outputs[0], (float *)inputs[0], n, m.valueToAdd);
    } else if (m.dataType == nvinfer1::DataType::kHALF) {
        std::cout << "Running fp16 kernel" << std::endl;
        std::this_thread::sleep_for(10ms);
        AddValue<<<(n + 1023) / 1024, 1024>>>((__half *)outputs[0], (__half *)inputs[0], n, (__half)m.valueToAdd);
    } else {
        std::cout << "Running int8 kernel" << std::endl;
        std::this_thread::sleep_for(0ms);
        float valueToAdd = m.valueToAdd / m.scale;
        std::cout << "valueToAdd (int8 scaled): " << valueToAdd << ", " << (int)valueToAdd << std::endl;
        AddValue<<<(n + 1023) / 1024, 1024>>>((int8_t *)outputs[0], (int8_t *)inputs[0], n, (int8_t)valueToAdd);
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(AddPluginCreator);
