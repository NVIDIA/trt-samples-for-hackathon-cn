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

#include <cub/cub.cuh>
#include <cuda.h>
#include <curand.h>
#include <stdlib.h>

const int globalNRow = 16;
const int globalNCol = 192;

__global__ void distribution192(float *pInDevice, float *pSample, int *pIndexDevice, float *pLossDevice, int *compareList)
{
    const int n  = globalNCol; // share memory 放不下
    int       bx = blockIdx.x, tx = threadIdx.x;

    typedef cub::BlockScan<float, n>           BlockScan;
    __shared__ typename BlockScan::TempStorage tempScan;
    float &                                    tDataScan = pInDevice[bx * n + tx];
    BlockScan(tempScan).InclusiveSum(tDataScan, tDataScan);

    pInDevice[bx * n + tx] /= pInDevice[bx * n + n - 1];
    __syncthreads();

    compareList[bx * n + tx] = int(pSample[bx] >= tDataScan);
    __syncthreads();

    typedef cub::BlockReduce<int, n>             BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempReduce;
    int &                                        tDataReduce = compareList[bx * n + tx];
    int                                          index       = BlockReduce(tempReduce).Sum(tDataReduce);

    if (tx == 0)
    {
        pIndexDevice[bx] = index;
        pLossDevice[bx]  = -__logf((index == 0) ? pInDevice[bx * n] : (pInDevice[bx * n + index] - pInDevice[bx * n + index - 1]));
        //pLossDevice[bx] = -__logf(pInDevice[bx*n+int(index)]-pInDevice[bx*n+int(index)-1]);
        //printf("%3d,%3d,%f,%f\n", bx, int(index), pInDevice[bx*n+int(index)],pInDevice[bx*n+int(index)-1]);
    }
    return;
}

int main()
{
    int    nRow = globalNRow, nCol = globalNCol;
    float *pInHost, *pInDevice, *pLossHost, *pLossDevice, *pSample;
    int *  pIndexHost, *pIndexDevice, *pCompareList;

    pInHost    = (float *)malloc(nRow * nCol * sizeof(float));
    pLossHost  = (float *)malloc(nRow * sizeof(float));
    pIndexHost = (int *)malloc(nRow * sizeof(float));
    cudaMalloc((void **)&pInDevice, nRow * nCol * sizeof(float));
    cudaMalloc((void **)&pLossDevice, nRow * sizeof(float));
    cudaMalloc((void **)&pIndexDevice, nRow * sizeof(float));
    cudaMalloc((void **)&pSample, nRow * sizeof(float));
    cudaMalloc((void **)&pCompareList, nRow * nCol * sizeof(int));

    srand(97);
    for (int i = 0; i < nRow * nCol; ++i)
    {
        float temp = float(rand()) / RAND_MAX;
        pInHost[i] = temp * temp;
    }

    cudaMemcpy(pInDevice, pInHost, nRow * nCol * sizeof(float), cudaMemcpyHostToDevice);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
    curandSetPseudoRandomGeneratorSeed(gen, 97ULL);
    curandGenerateUniform(gen, pSample, nRow);

    distribution192<<<nRow, nCol>>>(pInDevice, pSample, pIndexDevice, pLossDevice, pCompareList);

    cudaMemcpy(pIndexHost, pIndexDevice, nRow * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pLossHost, pLossDevice, nRow * sizeof(float), cudaMemcpyDeviceToHost);

    int sum = 0;
    for (int i = 0; i < nRow; ++i)
    {
        printf("%3d -> %3d\t%.4f\n", i, pIndexHost[i], pLossHost[i]);
        sum += pIndexHost[i];
    }
    printf("%f\n", (float)sum / nRow);

    curandDestroyGenerator(gen);
    free(pInHost);
    free(pIndexHost);
    free(pLossHost);
    cudaFree(pInDevice);
    cudaFree(pIndexDevice);
    cudaFree(pLossDevice);
    cudaFree(pSample);
}
