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

#include "ReducePlugin.h"

__global__ void reduce2(float *input, float *output)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x, index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float     a = input[index], b = input[index + blockDim.x];
    output[id] = max(a, b);
    return;
}

template<int n>
__global__ void reduceN(float *input, float *output)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x, index = blockIdx.x * blockDim.x * n + threadIdx.x, stride = blockDim.x;
    float     res = MIN_FLOAT;
    for (int i = 0; i < n; ++i)
        res = max(res, input[index + i * stride]);
    output[id] = res;
    return;
}

template<int n>
__global__ void reduceNSum(float *input, float *output)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x, index = blockIdx.x * blockDim.x * n + threadIdx.x, stride = blockDim.x;
    float     res = 0;
    for (int i = 0; i < n; ++i)
        res += input[index + i * stride];
    output[id] = res;
    return;
}

__global__ void reduce2Half(half *input, half *output)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x, index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    half      a = input[index], b = input[index + blockDim.x];
    output[id] = __hgt(a, b) ? a : b;
    return;
}

template<int n>
__global__ void reduceNHalf(half *input, half *output)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x, index = blockIdx.x * blockDim.x * n + threadIdx.x, stride = blockDim.x;
    half      res = MIN_FLOAT;
    for (int i = 0; i < n; ++i)
    {
        half temp = input[index + i * stride];
        res       = __hgt(res, temp) ? res : temp;
    }
    output[id] = res;
    return;
}

template<int n>
__global__ void reduceNSumHalf(half *input, half *output)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x, index = blockIdx.x * blockDim.x * n + threadIdx.x, stride = blockDim.x;
    half      res = 0;
    for (int i = 0; i < n; ++i)
        res = __hadd(res, input[index + i * stride]);
    output[id] = res;
    return;
}

int ReducePlugin::enqueue(int batchSize, const void *const *input, void **output, void *workspace, cudaStream_t stream)
{
    //printf(">%d,%d,%d,%d\n",m.nRow,m.nCol,m.nReduce,m.isFp16);
    if (m.isFp16)
    {
        switch (m.nReduce)
        {
        case 2:
            reduce2Half<<<batchSize * m.nRow, m.nCol, 0, stream>>>((half *)input[0], (half *)output[0]);
            break;
        case 5:
            (reduceNHalf<5>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((half *)input[0], (half *)output[0]);
            break;
        case 6:
            (reduceNHalf<6>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((half *)input[0], (half *)output[0]);
            break;
        case 10:
            (reduceNHalf<10>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((half *)input[0], (half *)output[0]);
            break;
        case 15:
            (reduceNHalf<15>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((half *)input[0], (half *)output[0]);
            break;
        case 16:
            (reduceNHalf<16>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((half *)input[0], (half *)output[0]);
            break;
        case 30:
        {
            if (m.isSum)
                (reduceNSumHalf<30>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((half *)input[0], (half *)output[0]);
            else
                (reduceNHalf<30>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((half *)input[0], (half *)output[0]);
            break;
        }
        case 82:
            (reduceNHalf<82>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((half *)input[0], (half *)output[0]);
            break;
        default:
            printf("Failed matching m.nReduce == %d in Fp16\n", m.nCol);
        }
    }
    else
    {
        switch (m.nReduce)
        {
        case 2:
            reduce2<<<batchSize * m.nRow, m.nCol, 0, stream>>>((float *)input[0], (float *)output[0]);
            break;
        case 5:
            (reduceN<5>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((float *)input[0], (float *)output[0]);
            break;
        case 6:
            (reduceN<6>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((float *)input[0], (float *)output[0]);
            break;
        case 10:
            (reduceN<10>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((float *)input[0], (float *)output[0]);
            break;
        case 15:
            (reduceN<15>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((float *)input[0], (float *)output[0]);
            break;
        case 16:
            (reduceN<16>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((float *)input[0], (float *)output[0]);
            break;
        case 30:
        {
            if (m.isSum)
                (reduceNSum<30>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((float *)input[0], (float *)output[0]);
            else
                (reduceN<30>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((float *)input[0], (float *)output[0]);
            break;
        }
        case 82:
            (reduceN<82>)<<<batchSize * m.nRow, m.nCol, 0, stream>>>((float *)input[0], (float *)output[0]);
            break;
        default:
            printf("Failed matching m.nReduce == %d in Fp32\n", m.nCol);
        }
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(ReducePluginCreator);
