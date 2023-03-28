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

#include "MaskPlugin.h"

using namespace nvinfer1;
using namespace plugin;

PluginFieldCollection    MaskPluginCreator::fc_ {};
std::vector<PluginField> MaskPluginCreator::attr_;

template<typename T>
__global__ void mask2DPluginKernel(T *output0, T *output1, T *output2, int *input1, int nSL)
{
    const int nBlockPerCol = gridDim.x;
    const int iBatch       = blockIdx.y / nBlockPerCol;
    const int validWidth   = input1[iBatch];
    const int indexX       = blockIdx.x * blockDim.x + threadIdx.x;
    const int indexY       = blockIdx.y % nBlockPerCol * blockDim.y + threadIdx.y;
    if (indexX < nSL && indexY < nSL)
    {
        T value0 = (indexX < validWidth && indexY < validWidth) ? T(1) : T(0);
        T value1 = (indexX < validWidth && indexY < validWidth) ? T(0) : negtiveInfinity<T>();

        output0[iBatch * nSL * nSL * 4 + indexY * nSL + indexX]                 = value0;
        output0[iBatch * nSL * nSL * 4 + nSL * nSL * 1 + indexY * nSL + indexX] = value0;
        output0[iBatch * nSL * nSL * 4 + nSL * nSL * 2 + indexY * nSL + indexX] = value0;
        output0[iBatch * nSL * nSL * 4 + nSL * nSL * 3 + indexY * nSL + indexX] = value0;

        output1[iBatch * nSL * nSL * 4 + indexY * nSL + indexX]                 = value1;
        output1[iBatch * nSL * nSL * 4 + nSL * nSL * 1 + indexY * nSL + indexX] = value1;
        output1[iBatch * nSL * nSL * 4 + nSL * nSL * 2 + indexY * nSL + indexX] = value1;
        output1[iBatch * nSL * nSL * 4 + nSL * nSL * 3 + indexY * nSL + indexX] = value1;
    }

    if (indexY < nSL)
    {
        T value0 = (indexY < validWidth) ? T(1) : T(0);

        for (int i = 0; i < 10; ++i)
        {
            output2[iBatch * nSL * 320 + indexY * 320 + i * 32 + threadIdx.x] = value0;
        }
        // if nsL > 320, we can unroll the for-loop as:
        // if(blockIdx.x < 10)
        //     output2[iBatch * nSL * 320 + indexY * 320 + blockIdx.x * 32 + threadIdx.x ] = value0;
    }

    return;
}

int32_t MaskPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    DEBUG_FUNC();
    const int nBS = inputDesc[0].dims.d[0], nSL = inputDesc[0].dims.d[1];
#if DEBUG_ENABLE
    printf("[MaskPlugin::enqueue]\n");
    printf("\tbatch_size = %d\n", nBS);
    printf("\tsequence_length = %d\n", nSL);

#endif

    dim3 grid(CEIL_DIVISION(nSL, WARP_SIZE), nBS * CEIL_DIVISION(nSL, WARP_SIZE)), block(WARP_SIZE, WARP_SIZE);
    if (inputDesc[0].type == DataType::kHALF)
    {
        (mask2DPluginKernel<half>)<<<grid, block, 0, stream>>>((half *)outputs[0], (half *)outputs[1], (half *)outputs[2], (int *)inputs[1], nSL);
    }
    else
    {
        (mask2DPluginKernel<float>)<<<grid, block, 0, stream>>>((float *)outputs[0], (float *)outputs[1], (float *)outputs[2], (int *)inputs[1], nSL);
    }

    return 0;
}

REGISTER_TENSORRT_PLUGIN(MaskPluginCreator);
