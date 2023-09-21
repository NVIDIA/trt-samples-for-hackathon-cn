/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cub/cub.cuh>
#include "common/checkMacrosPlugin.h"
#include "gatherV2Plugin.h"

#define CUDA_NUM_THREADS 512

// CUDA: number of blocks for threads.
#define CUDA_GET_BLOCKS(N) (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

using nvinfer1::plugin::GatherV2PluginCreator;
using nvinfer1::plugin::GatherV2Plugin;

template<typename T>
__global__ void gatherV2Kernel(const int nbThreads, T const* in, T* out, int const* indices, int seqLen) {
    CUDA_KERNEL_LOOP(idx, nbThreads) {
        int di = idx / seqLen;
        int dk = idx % seqLen;  // [0, 768)
        int ind = indices[di];
        if (ind < 0 || ind >= 49408)
            return;
        out[di * seqLen + dk] = in[ind * seqLen + dk];
    }
}

int32_t GatherV2Plugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {

    nvinfer1::DataType dataType = inputDesc[0].type;
    PLUGIN_ASSERT(inputDesc[0].type == DataType::kFLOAT);
    PLUGIN_ASSERT(inputDesc[1].type == DataType::kINT32);
    PLUGIN_ASSERT(outputDesc[0].type == DataType::kFLOAT);
    PLUGIN_ASSERT(inputDesc[0].dims.nbDims == 2);  // 49408x768
    PLUGIN_ASSERT(inputDesc[1].dims.nbDims == 2);  // 2x77
    PLUGIN_ASSERT(outputDesc[0].dims.nbDims == 3);  // 2x77x768
    PLUGIN_ASSERT(inputDesc[0].dims.d[0] == 49408);
    PLUGIN_ASSERT(inputDesc[0].dims.d[1] == 768);
    PLUGIN_ASSERT(inputDesc[1].dims.d[0] == 2);
    PLUGIN_ASSERT(inputDesc[1].dims.d[1] == 77);
    PLUGIN_ASSERT(outputDesc[0].dims.d[0] == 2);
    PLUGIN_ASSERT(outputDesc[0].dims.d[1] == 77);
    PLUGIN_ASSERT(outputDesc[0].dims.d[2] == 768);

    int bs = inputDesc[1].dims.d[0];  // 2
    int num_indices = inputDesc[1].dims.d[1];  // 77
    int seqLen = inputDesc[0].dims.d[1];  // 768
    const int32_t nbThreads = bs*num_indices*seqLen;
    switch (dataType) {
        case DataType::kFLOAT: {
            auto* input = static_cast<const float*>(inputs[0]);
            auto* indices = static_cast<int const*>(inputs[1]);
            auto* out = static_cast<float*>(outputs[0]);
            gatherV2Kernel<float> <<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS, 0, stream>>>(nbThreads, input, out, indices, seqLen);
            break;
        } case DataType::kHALF: {
            auto* input = static_cast<half const*>(inputs[0]);
            auto* indices = static_cast<int const*>(inputs[1]);
            auto* out = static_cast<half*>(outputs[0]);
            gatherV2Kernel<half> <<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS, 0, stream>>>(nbThreads, input, out, indices, seqLen);
            break;
        } case DataType::kINT8: {
            auto* input = static_cast<const int8_t*>(inputs[0]);
            auto* indices = static_cast<const int32_t*>(inputs[1]);
            auto* out = static_cast<const int8_t*>(outputs[0]);
            break;
        } default: PLUGIN_ASSERT(false)
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(GatherV2PluginCreator);
