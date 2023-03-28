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

#include "Mask2DPlugin.h"

template<typename T>
__global__ void mask2DPluginKernel(int *lod0, int *lod1, T *output, int nGroup, int nHeight, int nWidth, T mask2DTrueValue, T mask2DFalseValue)
{
    const int nYBlockPerGroup = gridDim.y / nGroup;
    const int indexG          = blockIdx.y / nYBlockPerGroup;                   // 线程所处 group 序号
    const int validHeight = lod0[indexG], validWidth = lod1[indexG];            // 线程所处 group 在行列方向上的宽度
    const int indexY = blockIdx.y % nYBlockPerGroup * blockDim.y + threadIdx.y; // 线程在 group 内的行列偏移
    const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("(%d,%d,%d,%d)->(%d,%d,%d,%d,%d,%d)->(%d,%d)\n", blockIdx.y,blockIdx.x,threadIdx.y,threadIdx.x,
    //                                                        indexG,indexG*nHeight*nWidth,validHeight,validWidth,indexY,indexX,
    //                                                        indexG * nHeight * nWidth + indexY * nWidth + indexX, indexY < lod0[indexG] && indexX < lod1[indexG] );

    if (indexY >= nHeight || indexX >= nWidth) // 超出数组边界的线程不做任何事
        return;
    output[indexG * nHeight * nWidth + indexY * nWidth + indexX] = (indexY < validHeight && indexX < validWidth) ? mask2DTrueValue : mask2DFalseValue;
    return;
}

int Mask2DPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream)
{
    dim3 grid(CEIL_DIVISION(m.nWidth, WARP_SIZE), m.nGroup * CEIL_DIVISION(m.nHeight, WARP_SIZE)), block(min(m.nWidth, WARP_SIZE), min(m.nHeight, WARP_SIZE));
    //printf("d=%2d,g=%2d,h=%2d,w=%2d,grid=(%d,%d),block=(%d,%d)\n",m.datatype,m.nGroup,m.nHeight,m.nWidth,grid.x,grid.y,block.x,block.y);
    switch (m.datatype)
    {
    case 0:
        mask2DPluginKernel<float><<<grid, block, 0, stream>>>((int *)inputs[1], (int *)inputs[2], (float *)outputs[0], m.nGroup, m.nHeight, m.nWidth, m.mask2DTrueValue, m.mask2DFalseValue);
        break;
    case 1:
        mask2DPluginKernel<__half><<<grid, block, 0, stream>>>((int *)inputs[1], (int *)inputs[2], (__half *)outputs[0], m.nGroup, m.nHeight, m.nWidth, __float2half(m.mask2DTrueValue), __float2half((m.mask2DFalseValue)));
        break;
    default:
        printf("[Mask2DPlugin::enqueue] Error datatype!\n");
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(Mask2DPluginCreator);
