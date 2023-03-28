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

#include "cub/cub.cuh"

#include <cuda.h>
#include <curand.h>
#include <stdlib.h>

const int nGlobalRow      = 16;
const int nGlobalColSmall = 32;
const int nGlobalColLarge = 128;

template<int n>
__global__ void sampleSmallKernel(float *pDeviceProbabilityColumn, float *pTargetRandomValue, int *pDeviceIndex, float *pDeviceEntropy)
{
    const int        bx = blockIdx.x, tx = threadIdx.x;
    __shared__ float probList[n]; // 一行一个分布列
    probList[tx] = pDeviceProbabilityColumn[bx * n + tx];
    typedef cub::WarpScan<float, n>           WarpScan; // 由概率分布列计算经验分布函数
    __shared__ typename WarpScan::TempStorage tempScan;
    float &                                   tDataScan = probList[tx];
    WarpScan(tempScan).InclusiveSum(tDataScan, tDataScan);

    tDataScan /= probList[n - 1]; // 若输入分布列没有归一化，则在这里除以闭前缀和的最后一个元素，以归一化
    //__syncthreads();
    //if(tx == 0)
    //printf("(%4d,%2d,%5d)\t%f\t%f\t%f\n",bx,tx,bx*n+tx, probList[0],probList[n/2], probList[n-1]);

    float sample = pTargetRandomValue[bx]; // sample ~ U[0,1]
    __syncthreads();

    __shared__ int pCompareList[n]; // 存放分布列一行的比较结果
    pCompareList[tx] = int(sample >= tDataScan);
    typedef cub::WarpReduce<int>                WarpReduce; // 找到首个累计概率大于 sample 的分布函数的下标，作为输出样本
    __shared__ typename WarpReduce::TempStorage tempReduce;
    int &                                       tDataReduce = pCompareList[tx];
    int                                         index       = min(WarpReduce(tempReduce).Sum(tDataReduce), n - 1);

    if (tx == 0) // 保存样本和交叉熵值
    {
        pDeviceIndex[bx]   = index;
        pDeviceEntropy[bx] = -__logf((index == 0) ? probList[0] : (probList[index] - probList[index - 1]));
        //printf("(%4d,%2d,%5d)\t%f\t%d\t%f\t%f\n",bx,tx,bx*n+tx,sample,index,probList[max(0,index-1], pDeviceEntropy[bx]);
    }
    return;
}

template<int n>
__global__ void sampleLargeKernel(float *pDeviceProbabilityColumn, float *pTargetRandomValue, int *pDeviceIndex, float *pDeviceEntropy, unsigned char *pCompareList)
{
    const int bx = blockIdx.x, tx = threadIdx.x;

    typedef cub::BlockScan<float, n>           BlockScan;
    __shared__ typename BlockScan::TempStorage tempScan;
    float &                                    tDataScan = pDeviceProbabilityColumn[bx * n + tx];
    BlockScan(tempScan).InclusiveSum(tDataScan, tDataScan);
    __syncthreads(); // 必须同步

    pDeviceProbabilityColumn[bx * n + tx] /= pDeviceProbabilityColumn[bx * n + n - 1];
    __syncthreads();

    pCompareList[bx * n + tx] = (unsigned char)int(pTargetRandomValue[bx] >= tDataScan);
    __syncthreads();

    typedef cub::BlockReduce<unsigned char, n>   BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempReduce;
    unsigned char &                              tDataReduce = pCompareList[bx * n + tx];
    unsigned char                                index       = (unsigned char)min(BlockReduce(tempReduce).Sum(tDataReduce), n - 1);
    __syncthreads();

    if (tx == 0)
    {
        pDeviceIndex[bx]   = index;
        pDeviceEntropy[bx] = -__logf((index == 0) ? pDeviceProbabilityColumn[bx * n] : (pDeviceProbabilityColumn[bx * n + index] - pDeviceProbabilityColumn[bx * n + index - 1]));
        //printf("(%4d,%2d,%5d)\t%f\t%d\t%f\t%f\n",bx,tx,bx*n+tx,sample,index,probList[max(bx*n,bx*n+index-1], pDeviceEntropy[bx]);
    }
    return;
}

int sampleSmall()
{
    const int nRow = nGlobalRow, nCol = nGlobalColSmall;
    float *   pHostProbabilityColumn, *pDeviceProbabilityColumn, *pHostEntropy, *pDeviceEntropy, *pTargetRandomValue;
    int *     pHostIndex, *pDeviceIndex;

    pHostProbabilityColumn = (float *)malloc(nRow * nCol * sizeof(float));
    pHostIndex             = (int *)malloc(nRow * sizeof(float));
    pHostEntropy           = (float *)malloc(nRow * sizeof(float));
    cudaMalloc((void **)&pDeviceProbabilityColumn, nRow * nCol * sizeof(float));
    cudaMalloc((void **)&pDeviceEntropy, nRow * sizeof(float));
    cudaMalloc((void **)&pDeviceIndex, nRow * sizeof(float));
    cudaMalloc((void **)&pTargetRandomValue, nRow * sizeof(float));

    srand(97);
    for (int i = 0; i < nRow * nCol; ++i)
    {
        float temp                = float(rand()) / RAND_MAX;
        pHostProbabilityColumn[i] = temp;
    }

    cudaMemcpy(pDeviceProbabilityColumn, pHostProbabilityColumn, sizeof(float) * nRow * nCol, cudaMemcpyHostToDevice);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
    curandSetPseudoRandomGeneratorSeed(gen, 97ULL);
    curandGenerateUniform(gen, pTargetRandomValue, nRow);

    (sampleSmallKernel<nCol>)<<<nRow, nCol>>>(pDeviceProbabilityColumn, pTargetRandomValue, pDeviceIndex, pDeviceEntropy);

    cudaMemcpy(pHostIndex, pDeviceIndex, sizeof(int) * nRow, cudaMemcpyDeviceToHost);
    cudaMemcpy(pHostEntropy, pDeviceEntropy, sizeof(float) * nRow, cudaMemcpyDeviceToHost);

    printf("row   indx  entropy\n");
    for (int i = 0; i < nRow; ++i)
    {
        printf("%3d -> %3d, %.4f\n", i, pHostIndex[i], pHostEntropy[i]);
    }

    curandDestroyGenerator(gen);
    free(pHostProbabilityColumn);
    free(pHostIndex);
    free(pHostEntropy);
    cudaFree(pDeviceProbabilityColumn);
    cudaFree(pDeviceIndex);
    cudaFree(pDeviceEntropy);
    cudaFree(pTargetRandomValue);
    return 0;
}

int sampleLarge()
{
    const int      nRow = nGlobalRow, nCol = nGlobalColLarge;
    float *        pHostProbabilityColumn, *pDeviceProbabilityColumn, *pHostEntropy, *pDeviceEntropy, *pTargetRandomValue;
    int *          pHostIndex, *pDeviceIndex;
    unsigned char *pCompareList;

    pHostProbabilityColumn = (float *)malloc(nRow * nCol * sizeof(float));
    pHostIndex             = (int *)malloc(nRow * sizeof(float));
    pHostEntropy           = (float *)malloc(nRow * sizeof(float));
    cudaMalloc((void **)&pDeviceProbabilityColumn, nRow * nCol * sizeof(float));
    cudaMalloc((void **)&pDeviceEntropy, nRow * sizeof(float));
    cudaMalloc((void **)&pDeviceIndex, nRow * sizeof(float));
    cudaMalloc((void **)&pTargetRandomValue, nRow * sizeof(float));
    cudaMalloc((void **)&pCompareList, nRow * sizeof(unsigned char));

    srand(97);
    for (int i = 0; i < nRow * nCol; ++i)
    {
        float temp                = float(rand()) / RAND_MAX;
        pHostProbabilityColumn[i] = temp;
    }

    cudaMemcpy(pDeviceProbabilityColumn, pHostProbabilityColumn, sizeof(float) * nRow * nCol, cudaMemcpyHostToDevice);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
    curandSetPseudoRandomGeneratorSeed(gen, 97ULL);
    curandGenerateUniform(gen, pTargetRandomValue, nRow);

    (sampleLargeKernel<nCol>)<<<nRow, nCol>>>(pDeviceProbabilityColumn, pTargetRandomValue, pDeviceIndex, pDeviceEntropy, pCompareList);

    cudaMemcpy(pHostIndex, pDeviceIndex, sizeof(int) * nRow, cudaMemcpyDeviceToHost);
    cudaMemcpy(pHostEntropy, pDeviceEntropy, sizeof(float) * nRow, cudaMemcpyDeviceToHost);

    printf("row    idx  entropy\n");
    for (int i = 0; i < nRow; ++i)
    {
        printf("%3d -> %3d, %.4f\n", i, pHostIndex[i], pHostEntropy[i]);
    }

    curandDestroyGenerator(gen);
    free(pHostProbabilityColumn);
    free(pHostIndex);
    free(pHostEntropy);
    cudaFree(pDeviceProbabilityColumn);
    cudaFree(pDeviceIndex);
    cudaFree(pDeviceEntropy);
    cudaFree(pTargetRandomValue);
    cudaFree(pCompareList);
    return 0;
}

int main()
{
    sampleSmall();
    sampleLarge();
    return 0;
}
