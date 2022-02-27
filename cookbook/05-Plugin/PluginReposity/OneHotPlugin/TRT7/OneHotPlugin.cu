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

__global__ void oneHot(int *input, float *output)
{
    const int bx = blockIdx.x, tx = threadIdx.x, index = input[bx];
    output[bx * blockDim.x + tx] = ((tx == index) ? 1.0f : 0.0f);
    return;
}

__global__ void oneHotLarge(int *input, float *output, int nEmbed)
{
    const int bx = blockIdx.x, tx = threadIdx.x, index = input[bx];
    for(int i = 0; i < CEIL(nEmbed,blockDim.x); i++)
    {
        int tx2 = tx + i * blockDim.x;
        if(tx2 >= nEmbed)
            return;
        output[bx * nEmbed + tx2] = ((tx2 == index) ? 1.0f : 0.0f);
    }
    return;
}

__global__ void oneHotHalf(int *input, __half2 *output)
{
    const int bx = blockIdx.x, tx = threadIdx.x, index = input[bx];
    __half2 value = __floats2half2_rn(0.0f,0.0f);
    if(tx == index>>1)
        value = __floats2half2_rn( 1.0f-float(index & 1),float(index & 1) );// tx serves position of 'tx*2' and 'tx*2+1', and value is (0,1) if index odd, (1,0) if index even
    output[bx * blockDim.x + tx ] = value;
    return;
}

__global__ void oneHotHalfLarge(int *input, __half2 *output, int nEmbed)
{
    const int bx = blockIdx.x, tx = threadIdx.x, index = input[bx], nHalfPerRow = nEmbed>>1;
    for(int i = 0; i < CEIL(nHalfPerRow,blockDim.x); i++)
    {
        int tx2 = tx + i * blockDim.x;
        if(tx2 >= nHalfPerRow)
            return;
        __half2 value = __floats2half2_rn(0.0f,0.0f);
        if(tx2 == index>>1)
            value = __floats2half2_rn( 1.0f-float(index & 1),float(index & 1) );
        output[bx * nHalfPerRow + tx2] = value;
    }
    return;
}

int OneHotPlugin::enqueue(int batchSize, const void * const *input, void **output, void* workspace, cudaStream_t stream)
{
    if(m.nEmbed > 1024)
    {
        if(m.isFp16)
            oneHotHalfLarge <<< batchSize * m.nRow, 512, 0, stream>>> ((int*)input[0], (__half2*)output[0], m.nEmbed);
        else
            oneHotLarge <<< batchSize * m.nRow, 1024, 0, stream>>> ((int*)input[0], (float*)output[0], m.nEmbed);
    }
    else
    {
        if(m.isFp16)
            oneHotHalf <<< batchSize * m.nRow, m.nEmbed >>1, 0, stream>>> ((int*)input[0], (__half2*)output[0]);
        else
            oneHot <<< batchSize * m.nRow, m.nEmbed, 0, stream>>> ((int*)input[0], (float*)output[0]);
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(OneHotPluginCreator);

