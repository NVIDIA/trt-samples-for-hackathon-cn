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
__global__ void distribution(float *pInDevice, float *pSample, int *pIndexDevice, float *pLossDevice)
{
    int bx = blockIdx.x, tx = threadIdx.x;    

    __shared__ float probList[n];                                                                   // 存放分布列的一行（一个分量的分布）
    probList[tx] = pInDevice[bx * n + tx];
    typedef cub::WarpScan<float, n> WarpScan;                                                       // 由概率分布列计算经验分布函数
    __shared__ typename WarpScan::TempStorage tempScan;
    float &tDataScan = probList[tx];
    WarpScan(tempScan).InclusiveSum(tDataScan, tDataScan);

    tDataScan /= probList[n-1];                                                                     // 若输入分布列没有归一化，则在这里除以闭前缀和的最后一个元素，以归一化
    //if(tx == 0)
    //    printf("(%4d,%2d,%5d)\t%f\t%f\t%f\n",bx,tx,bx*n+tx, probList[0],probList[n/2], probList[n-1]);

    float sample = pSample[bx];                                                                     // sample ~ U[0,1]

    __shared__ int compareList[n];                                                                  // 存放分布列一行的比较结果
    compareList[tx] = int(sample >= tDataScan);
    typedef cub::WarpReduce<int> WarpReduce;                                                        // 找到首个累计概率大于 sample 的分布函数的下标，作为输出样本
    __shared__ typename WarpReduce::TempStorage tempReduce;
    int &tDataReduce = compareList[tx];
    int index = WarpReduce(tempReduce).Sum(tDataReduce);

    if(tx == 0)                                                                                     // 保存样本和交叉熵值
    {
        pIndexDevice[bx] = index;
        pLossDevice[bx] = -__logf( (index==0) ? probList[0]:(probList[index]-probList[index-1]) );
        //printf("(%4d,%2d,%5d)\t%f\t%d\t%f\t%f\n",bx,tx,bx*n+tx, sample,index,probList[max(0,index-1)],probList[index]);
    }
    return;
}

__global__ void distribution192(float *pInDevice, float *pSample, int *pIndexDevice, float *pLossDevice, unsigned char *compareList)
{
    const int n = 192;                                                                              // share memory 放不下，单独写
    int bx = blockIdx.x, tx = threadIdx.x;

    typedef cub::BlockScan<float, n> BlockScan;
    __shared__ typename BlockScan::TempStorage tempScan;
    float &tDataScan = pInDevice[bx * n + tx];
    BlockScan(tempScan).InclusiveSum(tDataScan, tDataScan);
    __syncthreads();                                                                                // 必须同步

    pInDevice[bx * n + tx] /= pInDevice[bx * n + n-1];

    compareList[bx * n + tx] = (unsigned char)int(pSample[bx] >= tDataScan);
    typedef cub::BlockReduce<unsigned char, n> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempReduce;
    unsigned char &tDataReduce = compareList[bx * n + tx];
    unsigned char index = BlockReduce(tempReduce).Sum(tDataReduce);
    //__syncthreads();                                                                                // 可以不用同步？
    
    if(tx == 0)
    {
        pIndexDevice[bx] = int(index);
        pLossDevice[bx] = -__logf( (index==0) ? pInDevice[bx*n]:(pInDevice[bx*n+index]-pInDevice[bx*n+index-1]) );
    }
    return;
}

int RandomPlugin::enqueue(int batchSize, const void * const *input, void **output, void* workspace, cudaStream_t stream)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
    curandSetPseudoRandomGeneratorSeed(gen, m.seed);
    curandGenerateUniform(gen, (float*)workspace, m.nRow);
    
    //printf("m -> (%d,%d)\n", m.nRow, m.nCol);
    switch(m.nCol)
    {
    case 4:
        (distribution<4>) <<< batchSize * m.nRow, m.nCol, 0, stream>>> ((float*)input[0], (float*)workspace, (int*)output[0], (float*)output[1]); break;
    case 9:
        (distribution<9>) <<< batchSize * m.nRow, m.nCol, 0, stream>>> ((float*)input[0], (float*)workspace, (int*)output[0], (float*)output[1]); break;
    case 30:
        (distribution<30>) <<< batchSize * m.nRow, m.nCol, 0, stream>>> ((float*)input[0], (float*)workspace, (int*)output[0], (float*)output[1]); break;
    case 192:
        {
            unsigned char *compareList = (unsigned char*)((char*)workspace + ALIGNED(sizeof(float)*m.nRow));
            distribution192 <<< batchSize * m.nRow, m.nCol, 0, stream>>> ((float*)input[0], (float*)workspace, (int*)output[0], (float*)output[1], compareList);
            break;
        }
    default:
        break;//printf("Failed matching m.nCol with defiend kernel\n");
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(RandomPluginCreator);

