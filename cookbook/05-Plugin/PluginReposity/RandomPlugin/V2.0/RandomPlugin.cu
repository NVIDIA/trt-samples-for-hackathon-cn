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

#include "RandomPlugin.h"

template <int n>
__global__ void distribution(float *input, float *pSample, int *outputIndex, float *outputValue)
{
    const int bx = blockIdx.x, tx = threadIdx.x;

    __shared__ float probList[n];                                                                   // 一行一个分布列
    probList[tx] = input[bx * n + tx];
    typedef cub::WarpScan<float, n> WarpScan;                                                       // 由概率分布列计算经验分布函数
    __shared__ typename WarpScan::TempStorage tempScan;
    float &tDataScan = probList[tx];
    WarpScan(tempScan).InclusiveSum(tDataScan, tDataScan);

    tDataScan /= probList[n-1];                                                                     // 若输入分布列没有归一化，则在这里除以闭前缀和的最后一个元素，以归一化
    //if(tx == 0)
        //printf("(%4d,%2d,%5d)\t%f\t%f\t%f\n",bx,tx,bx*n+tx, probList[0],probList[n/2], probList[n-1]);

    float sample = pSample[bx];                                                                     // sample ~ U[0,1]
    __shared__ int compareList[n];                                                                  // 存放分布列一行的比较结果
    compareList[tx] = int(sample >= tDataScan);
    typedef cub::WarpReduce<int> WarpReduce;                                                        // 找到首个累计概率大于 sample 的分布函数的下标，作为输出样本
    __shared__ typename WarpReduce::TempStorage tempReduce;
    int &tDataReduce = compareList[tx];
    int index = min(WarpReduce(tempReduce).Sum(tDataReduce), n-1);

    if(tx == 0)                                                                                     // 保存样本和交叉熵值
    {
        outputIndex[bx] = index;
        outputValue[bx] = -__logf( (index==0) ? probList[0]:(probList[index]-probList[index-1]) );
        //printf("(%4d,%2d,%5d)\t%f\t%d\t%f\t%f\n",bx,tx,bx*n+tx, sample,index,probList[max(0,index-1)],probList[index]);
        //printf("%4d->%4d\t%f\n",
        //    bx,index,-__logf( (index==0) ? probList[0]:(probList[index]-probList[index-1]) ));
    }
    return;
}

__global__ void distribution192(float *input, float *pSample, int *outputIndex, float *outputValue, unsigned char *compareList)
{
    const int bx = blockIdx.x, tx = threadIdx.x, n = 192;                                           // share memory 放不下，单独写

    typedef cub::BlockScan<float, n> BlockScan;
    __shared__ typename BlockScan::TempStorage tempScan;
    float &tDataScan = input[bx * n + tx];
    BlockScan(tempScan).InclusiveSum(tDataScan, tDataScan);
    __syncthreads();                                                                                // 必须同步

    input[bx * n + tx] /= input[bx * n + n-1];

    compareList[bx * n + tx] = (unsigned char)int(pSample[bx] >= tDataScan);
    __syncthreads();                                                                                // 可以不用同步？
    typedef cub::BlockReduce<unsigned char, n> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempReduce;
    unsigned char &tDataReduce = compareList[bx * n + tx];
    unsigned char index = (unsigned char)min(BlockReduce(tempReduce).Sum(tDataReduce), n-1);
    __syncthreads();                                                                                // 可以不用同步？

    if(tx == 0)
    {
        outputIndex[bx] = int(index);
        outputValue[bx] = -__logf( (index==0) ? input[bx*n]:(input[bx*n+index]-input[bx*n+index-1]) );
    }
    return;
}

template <int n>
__global__ void distributionHalf(__half *input, float *pSample, int *outputIndex, float *outputValue)
{
    const int bx = blockIdx.x, tx = threadIdx.x;

    __shared__ __half probList[n];                                                                    // 一行一个分布列
    probList[tx] = input[bx * n + tx];
    typedef cub::WarpScan<__half, n> WarpScan;                                                        // 由概率分布列计算经验分布函数
    __shared__ typename WarpScan::TempStorage tempScan;
    __half &tDataScan = probList[tx];
    WarpScan(tempScan).InclusiveSum(tDataScan, tDataScan);

    tDataScan /= probList[n-1];                                                                     // 若输入分布列没有归一化，则在这里除以闭前缀和的最后一个元素，以归一化
    //if(tx == 0)
        //printf("(%4d,%2d,%5d)\t%f\t%f\t%f\n",bx,tx,bx*n+tx, probList[0],probList[n/2], probList[n-1]);

    float sample = pSample[bx];                                                                     // sample ~ U[0,1]
    __shared__ int compareList[n];                                                                  // 存放分布列一行的比较结果
    compareList[tx] = int(sample >= __half2float(tDataScan));
    typedef cub::WarpReduce<int> WarpReduce;                                                        // 找到首个累计概率大于 sample 的分布函数的下标，作为输出样本
    __shared__ typename WarpReduce::TempStorage tempReduce;
    int &tDataReduce = compareList[tx];
    int index = min(WarpReduce(tempReduce).Sum(tDataReduce), n-1);

    if(tx == 0)                                                                                     // 保存样本和交叉熵值
    {
        outputIndex[bx] = index;
        outputValue[bx] = __half2float(-hlog( (index==0) ? probList[0]:(probList[index]-probList[index-1]) ));
        //printf("(%4d,%2d,%5d)\t%f\t%d\t%f\t%f\n",bx,tx,bx*n+tx, sample,index,probList[max(0,index-1)],probList[index]);
    }
    return;
}

__global__ void distributionHalf192(__half *input, float *pSample, int *outputIndex, float *outputValue, unsigned char *compareList)
{
    const int bx = blockIdx.x, tx = threadIdx.x, n = 192;                                           // share memory 放不下，单独写

    typedef cub::BlockScan<__half, n> BlockScan;
    __shared__ typename BlockScan::TempStorage tempScan;
    __half &tDataScan = input[bx * n + tx];
    BlockScan(tempScan).InclusiveSum(tDataScan, tDataScan);
    __syncthreads();                                                                                // 必须同步

    input[bx * n + tx] /= input[bx * n + n-1];

    compareList[bx * n + tx] = (unsigned char)int(pSample[bx] >= __half2float(tDataScan));
    __syncthreads();                                                                                // 可以不用同步？
    typedef cub::BlockReduce<unsigned char, n> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempReduce;
    unsigned char &tDataReduce = compareList[bx * n + tx];
    unsigned char index = (unsigned char)min(BlockReduce(tempReduce).Sum(tDataReduce), n-1);
    __syncthreads();                                                                                // 可以不用同步？

    if(tx == 0)
    {
        outputIndex[bx] = int(index);
        outputValue[bx] = __half2float(-hlog(
            (index==0) ? (input[bx*n]):(input[bx*n+index]-input[bx*n+index-1])
            ));

        printf("%4d->%4d\t%d\t%f\t%f\t%f\t%f\n",
            bx,index,int(index),
            __half2float(input[bx*n+index]),__half2float(input[bx*n+index-1]),
            __half2float((index==0) ? (input[bx*n]):(input[bx*n+index]-input[bx*n+index-1]))
        );

    }
    return;
}

int RandomPlugin::enqueue(int batchSize, const void * const *input, void **output, void* workspace, cudaStream_t stream)
{
    if(m.firstEnqueue)
    {
        curandSetStream(m.gen, stream);
        m.firstEnqueue = 0;
    }
    curandGenerateUniform(m.gen, (float*)workspace, batchSize * m.nRow);

    printf(">%d,%d,%d\n",m.nRow,m.nCol,m.isFp16);
    if(m.isFp16)
    {
        switch(m.nCol)
        {
        case 4:
            (distributionHalf<4>)   <<< batchSize * m.nRow, m.nCol, 0, stream>>> ((__half*)input[0], (float*)workspace, (int*)output[0], (float*)output[1]); break;
        case 9:
            (distributionHalf<9>)   <<< batchSize * m.nRow, m.nCol, 0, stream>>> ((__half*)input[0], (float*)workspace, (int*)output[0], (float*)output[1]); break;
        case 30:
            (distributionHalf<30>)  <<< batchSize * m.nRow, m.nCol, 0, stream>>> ((__half*)input[0], (float*)workspace, (int*)output[0], (float*)output[1]); break;
        case 192:
        {
            unsigned char *compareList = (unsigned char*)((char*)workspace + ALIGNED(batchSize * m.nRow * sizeof(float)));
            distributionHalf192 <<< batchSize * m.nRow, m.nCol, 0, stream>>> ((__half*)input[0], (float*)workspace, (int*)output[0], (float*)output[1], compareList);
            break;
        }
        default:
            printf("Failed matching m.nCol == %d in Fp16\n", m.nCol);
        }
    }
    else
    {
        switch(m.nCol)
        {
        case 4:
            (distribution<4>)   <<< batchSize * m.nRow, m.nCol, 0, stream>>> ((float*)input[0], (float*)workspace, (int*)output[0], (float*)output[1]); break;
        case 9:
            (distribution<9>)   <<< batchSize * m.nRow, m.nCol, 0, stream>>> ((float*)input[0], (float*)workspace, (int*)output[0], (float*)output[1]); break;
        case 30:
            (distribution<30>)  <<< batchSize * m.nRow, m.nCol, 0, stream>>> ((float*)input[0], (float*)workspace, (int*)output[0], (float*)output[1]); break;
        case 192:
        {
            unsigned char *compareList = (unsigned char*)((char*)workspace + ALIGNED(batchSize * m.nRow * sizeof(float)));
            distribution192 <<< batchSize * m.nRow, m.nCol, 0, stream>>> ((float*)input[0], (float*)workspace, (int*)output[0], (float*)output[1], compareList);
            break;
        }
        default:
            printf("Failed matching m.nCol == %d in Fp32\n", m.nCol);
        }
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(RandomPluginCreator);

