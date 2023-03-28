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

// An example of device radix sort by CUB
#define CUB_STDERR // print CUDA runtime error to console

#include <algorithm>
#include <cstdio>
#include <cub/device/device_radix_sort.cuh>

using namespace cub;

struct Pair
{
    float key;
    int   value;

    bool operator<(const Pair &b) const
    {
        if (key < b.key)
            return true;
        if (key > b.key)
            return false;

        unsigned int key_bits   = *reinterpret_cast<unsigned *>(const_cast<float *>(&key));
        unsigned int b_key_bits = *reinterpret_cast<unsigned *>(const_cast<float *>(&b.key));
        unsigned int HIGH_BIT   = 1u << 31;
        return ((key_bits & HIGH_BIT) != 0) && ((b_key_bits & HIGH_BIT) == 0); // true if key == -0 && b.key == +0
    }
};

int main()
{
    const int nItem     = 150;
    float *   hKey      = new float[nItem];
    float *   hKeyCPU   = new float[nItem];
    float *   hKeyGPU   = new float[nItem];
    int *     hValue    = new int[nItem];
    int *     hValueCPU = new int[nItem];
    int *     hValueGPU = new int[nItem];

    // initialization
    for (int i = 0; i < nItem; ++i)
    {
        hKey[i]   = (float)rand();
        hValue[i] = i;
    }

    // sort by CPU
    Pair *hPair = new Pair[nItem];
    for (int i = 0; i < nItem; ++i)
    {
        hPair[i].key   = hKey[i];
        hPair[i].value = hValue[i];
    }

    std::stable_sort(hPair, hPair + nItem);

    for (int i = 0; i < nItem; ++i)
    {
        hKeyCPU[i]   = hPair[i].key;
        hValueCPU[i] = hPair[i].value;
    }

    delete[] hPair;

    // sort by GPU
    DoubleBuffer<float> dKey; // special data structure used by CUB
    DoubleBuffer<int>   dValue;
    cudaMalloc((void **)&dKey.d_buffers[0], sizeof(float) * nItem);
    cudaMalloc((void **)&dKey.d_buffers[1], sizeof(float) * nItem);
    cudaMalloc((void **)&dValue.d_buffers[0], sizeof(int) * nItem);
    cudaMalloc((void **)&dValue.d_buffers[1], sizeof(int) * nItem);

    size_t tempByte = 0;
    void * dTemp    = NULL;
    DeviceRadixSort::SortPairs(dTemp, tempByte, dKey, dValue, nItem); // get temporary workspace, usually small but could not to be ignored
    cudaMalloc(&dTemp, tempByte);

    printf("before sort, dKey.selector = %d, dValue.selector = %d\n", dKey.selector, dValue.selector); // valid data locate in dKey[selector] and dValue[]
    cudaMemcpy(dKey.Current(), hKey, sizeof(float) * nItem, cudaMemcpyHostToDevice);
    cudaMemcpy(dValue.Current(), hValue, sizeof(int) * nItem, cudaMemcpyHostToDevice);

    DeviceRadixSort::SortPairs(dTemp, tempByte, dKey, dValue, nItem); // DeviceRadixSort::SortPairs, SortPairsDescending

    printf("after sort, dKey.selector = %d, dValue.selector = %d\n", dKey.selector, dValue.selector); // selector changed during the sort
    cudaMemcpy(hKeyGPU, dKey.Current(), sizeof(float) * nItem, cudaMemcpyDeviceToHost);
    cudaMemcpy(hValueGPU, dValue.Current(), sizeof(int) * nItem, cudaMemcpyDeviceToHost);

    // check result
    bool pass = true;
    for (int i = 0; i < nItem && pass == true; ++i)
    {
        if (hKeyCPU[i] != hKeyGPU[i])
        {
            printf("error at i = %d, hKeyCPU[i] = %f, hKeyGPU[i] = %f\n", i, hKeyCPU[i], hKeyGPU[i]);
            pass = false;
        }
        if (hValueCPU[i] != hValueGPU[i])
        {
            printf("error at i = %d, hValueCPU[i] = %d, hValueGPU[i] = %d\n", i, hValueCPU[i], hValueGPU[i]);
            pass = false;
        }
    }
    printf("Test %s\n", pass ? "succeed!" : "failed!");
    for (int i = 0; i < nItem; ++i)
    {
        printf("%3d: input(%.4E,%3d), outputCPU(%.4E,%3d), outputGPU(%.4E,%3d)\n",
               i,
               hKey[i],
               hValue[i],
               hKeyCPU[i],
               hValueCPU[i],
               hKeyGPU[i],
               hValueGPU[i]);
    }

    if (hKey)
        delete[] hKey;
    if (hKeyCPU)
        delete[] hKeyCPU;
    if (hKeyGPU)
        delete[] hKeyGPU;
    if (hValue)
        delete[] hValue;
    if (hValueCPU)
        delete[] hValueCPU;
    if (hValueGPU)
        delete[] hValueGPU;
    if (dKey.d_buffers[0])
        cudaFree(dKey.d_buffers[0]);
    if (dKey.d_buffers[1])
        cudaFree(dKey.d_buffers[1]);
    if (dValue.d_buffers[0])
        cudaFree(dValue.d_buffers[0]);
    if (dValue.d_buffers[1])
        cudaFree(dValue.d_buffers[1]);
    if (dTemp)
        cudaFree(dTemp);

    return 0;
}
