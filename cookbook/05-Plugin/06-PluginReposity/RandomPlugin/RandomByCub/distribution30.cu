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

#include<stdlib.h>
#include<cuda.h>
#include<curand.h>
#include "cub/cub.cuh"

const int globalNRow = 16;
const int globalNCol = 30;

__global__ void distribution30(float *pInDevice, float *pSample, int *pIndexDevice, float *pLossDevice)
{
    const int n = globalNCol;
    int bx = blockIdx.x, tx = threadIdx.x;

    typedef cub::WarpScan<float, n> WarpScan;                                                       // 由概率分布列计算经验分布函数
    __shared__ typename WarpScan::TempStorage tempScan;
    __shared__ float probList[n];
    probList[tx] = pInDevice[bx * n + tx];
    float &tDataScan = probList[tx];
    WarpScan(tempScan).InclusiveSum(tDataScan, tDataScan);

    tDataScan /= probList[n-1];                                                                     // 若输入分布列没有归一化，则在这里除以闭前缀和的最后一个元素，以归一化
    __syncthreads();
    //if(tx == 0)
    //    printf("(%4d,%2d,%5d)\t%f\t%f\n",bx,tx,id,tDataScan,probList[n-1]);

    float sample = pSample[bx];                                                                     // sample ~ U[0,1]

    typedef cub::WarpReduce<int> WarpReduce;                                                        // 找到首个累计概率大于 sample 的下标，作为样本值
    __shared__ typename WarpReduce::TempStorage tempReduce;
    __shared__ int compareList[n];
    compareList[tx] = int(sample >= tDataScan);
    __syncthreads();
    int &tDataReduce = compareList[tx];
    int index = WarpReduce(tempReduce).Sum(tDataReduce);
    if(tx == 0)
    {
        pIndexDevice[bx] = index;
        pLossDevice[bx] = -__logf( (index==0) ? probList[index]:(probList[index]-probList[index-1]) );
        //printf("(%4d,%2d,%5d)\t%f\t%d\t%f\t%f\n",bx,tx,bx*n+tx,sample,index,probList[index],
        //                                         -__logf( (index==0) ? probList[index]:(probList[index]-probList[index-1]) ) );
    }
    return;
}

int main()
{
    const int nRow = globalNRow, nCol = globalNCol;
    float *pInHost, *pInDevice, *pLossHost, *pLossDevice, *pSample;
    int   *pIndexHost, *pIndexDevice;

    pInHost     = (float *)malloc(nRow*nCol*sizeof(float));
    pLossHost   = (float *)malloc(nRow*sizeof(float));
    pIndexHost  = (int *)malloc(nRow*sizeof(float));
    cudaMalloc((void **)&pInDevice, nRow*nCol*sizeof(float));
    cudaMalloc((void **)&pLossDevice, nRow*sizeof(float));
    cudaMalloc((void **)&pIndexDevice, nRow*sizeof(float));
    cudaMalloc((void **)&pSample, nRow*sizeof(float));

    srand(97);
    for(int i = 0; i < nRow * nCol; i++)
    {
        float temp = float(rand()) / RAND_MAX;
        pInHost[i] = temp * temp;
    }

    cudaMemcpy(pInDevice, pInHost, nRow*nCol*sizeof(float), cudaMemcpyHostToDevice);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
    curandSetPseudoRandomGeneratorSeed(gen, 97ULL);
    curandGenerateUniform(gen, pSample, nRow);

    distribution30 <<< nRow, nCol >>> (pInDevice, pSample, pIndexDevice, pLossDevice);

    cudaMemcpy(pIndexHost, pIndexDevice, nRow * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pLossHost,  pLossDevice,  nRow * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i< nRow; i++)
        printf("%d -> %d, %.4f\n",i, pIndexHost[i], pLossHost[i]);

    curandDestroyGenerator(gen);
    free(pInHost);
    free(pIndexHost);
    free(pLossHost);
    cudaFree(pInDevice);
    cudaFree(pIndexDevice);
    cudaFree(pLossDevice);
    cudaFree(pSample);

}
