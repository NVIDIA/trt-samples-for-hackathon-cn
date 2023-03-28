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

#include "ReversePlugin.h"

template<typename T>
__global__ void reverse4Kernel(T *input, int *lod, T *output)
{
    const int nWidth = gridDim.x, nEmbed = blockDim.x, row = blockIdx.y, col = blockIdx.x, tx = threadIdx.x;
    int       src, dst, nValidWidth;
    T         value;

    nValidWidth = lod[row];
    if (col < nValidWidth)
    {
        src         = (row * nWidth + col) * nEmbed;
        value       = input[src + tx];
        dst         = (row * nWidth + nValidWidth - 1 - col) * nEmbed + tx;
        output[dst] = value;
    }
    return;
}

int ReversePlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream)
{
    //printf("t=%d, g=%d, w=%d, e=%d\n",m.datatype,m.nGroup,m.nWidth, m.nEmbed);
    switch (m.datatype)
    {
    case 0:
        reverse4Kernel<float><<<dim3(m.nWidth, m.nGroup), m.nEmbed, 0, stream>>>((float *)inputs[0], (int *)inputs[1], (float *)outputs[0]);
        break;
    case 1:
        reverse4Kernel<__half><<<dim3(m.nWidth, m.nGroup), m.nEmbed, 0, stream>>>((__half *)inputs[0], (int *)inputs[1], (__half *)outputs[0]);
        break;
    case 3:
        reverse4Kernel<int><<<dim3(m.nWidth, m.nGroup), m.nEmbed, 0, stream>>>((int *)inputs[0], (int *)inputs[1], (int *)outputs[0]);
        break;
    default:
        printf("[ReversePlugin::enqueue]Error datatype!\n");
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(ReversePluginCreator);
