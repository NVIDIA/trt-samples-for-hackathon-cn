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

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// 存放各矩阵维数的结构体
typedef struct
{
    unsigned int wa, ha, wb, hb, wc, hc;
} matrixSize;

int main()
{
    matrixSize ms;
    ms.wa = 3;
    ms.ha = 2;
    ms.wb = 4;
    ms.hb = ms.wa;
    ms.wc = ms.wb;
    ms.hc = ms.ha;

    unsigned int size_A     = ms.wa * ms.ha;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *      h_A        = (float *)malloc(mem_size_A);
    unsigned int size_B     = ms.wb * ms.hb;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *      h_B        = (float *)malloc(mem_size_B);
    unsigned int size_C     = ms.wc * ms.hc;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *      h_C        = (float *)malloc(mem_size_C);

    for (int i = 0; i < ms.ha * ms.wa; ++i)
        h_A[i] = i + 1;
    for (int i = 0; i < ms.hb * ms.wb; ++i)
        h_B[i] = i + 1;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, mem_size_A);
    cudaMalloc((void **)&d_B, mem_size_B);
    cudaMalloc((void **)&d_C, mem_size_C);
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    const float    alpha = 1.0f;
    const float    beta  = 0.0f;
    int            m = ms.ha, n = ms.wb, k = ms.wa;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    printf("\nA =\n");
    for (int i = 0; i < ms.ha * ms.wa; ++i)
    {
        printf("%3.2f\t", h_A[i]);
        if ((i + 1) % ms.wa == 0)
            printf("\n");
    }
    printf("\nB =\n");
    for (int i = 0; i < ms.hb * ms.wb; ++i)
    {
        printf("%3.2f\t", h_B[i]);
        if ((i + 1) % ms.wb == 0)
            printf("\n");
    }
    printf("\nC = A * B = \n");
    for (int i = 0; i < ms.hc * ms.wc; ++i)
    {
        printf("%3.2f\t", h_C[i]);
        if ((i + 1) % ms.wc == 0)
            printf("\n");
    }

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}
