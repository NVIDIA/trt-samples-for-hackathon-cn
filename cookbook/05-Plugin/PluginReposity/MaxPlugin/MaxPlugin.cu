/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "MaxPlugin.h"

template<typename T>
__global__ void maxKernel(T *input, int *lod, T *output, int nGroup, int totalWidth)
{
    const int validWidth = lod[blockIdx.x % nGroup], dst = blockIdx.x * blockDim.x + threadIdx.x;

    T temp, maxTemp = T(SMALL_NUMBER);
    for (int i = 0, src = blockIdx.x * totalWidth + threadIdx.x; i < validWidth; i++, src += blockDim.x)
    {
        temp    = input[src];
        maxTemp = (maxTemp > temp) ? maxTemp : temp;
    }
    output[dst] = maxTemp;
    return;
}

int MaxPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream)
{
    //printf("g=%2d,w=%2d,e=%2d\n",m.nGroup,m.nWidth,m.nEmbed);
    switch (m.datatype)
    {
    case 0:
        maxKernel<float><<<m.nGroup, m.nEmbed, 0, stream>>>((float *)inputs[0], (int *)inputs[1], (float *)outputs[0], m.nGroup, m.nWidth * m.nEmbed);
        break;
    case 1:
        maxKernel<__half><<<m.nGroup, m.nEmbed, 0, stream>>>((__half *)inputs[0], (int *)inputs[1], (__half *)outputs[0], m.nGroup, m.nWidth * m.nEmbed);
        break;
    default:
        //printf("[MaxPlugin::enqueue]Error datatype!\n");
        break;
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(MaxPluginCreator);
