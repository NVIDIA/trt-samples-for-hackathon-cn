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

#include "OneHotPlugin.h"

using namespace nvinfer1;
using namespace plugin;

PluginFieldCollection    OneHotPluginCreator::mFC {};
std::vector<PluginField> OneHotPluginCreator::mPluginAttributes;

template<typename T>
__global__ void OneHotPluginKernel(int *pArgmax, T *output, int nEmbed)
{
    const int tx = threadIdx.x, batch_id = blockIdx.x, index = blockIdx.y * blockDim.x + threadIdx.x;

    T value = T(0.0f);
    if (index < nEmbed)
    {
        if (pArgmax[batch_id] == index)
            value = T(1.0f);
        output[batch_id * nEmbed + index] = value;
    }
}

int OneHotPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    dim3 dimBlock, dimGrid;
    if (m.nEmbed > 1024)
    {
        dimBlock.x = 1024;
        dimGrid.y  = (m.nEmbed + dimBlock.x - 1) / dimBlock.x;
    }
    else
    {
        dimBlock.x = m.nEmbed;
        dimGrid.y  = 1;
    }
    dimGrid.x = inputDesc[0].dims.d[0] * m.nRow;

    if (m.isFp16)
        (OneHotPluginKernel<half>)<<<dimGrid, dimBlock, 0, stream>>>((int *)inputs[0], (half *)outputs[0], m.nEmbed);
    else
        (OneHotPluginKernel<float>)<<<dimGrid, dimBlock, 0, stream>>>((int *)inputs[0], (float *)outputs[0], m.nEmbed);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(OneHotPluginCreator);
