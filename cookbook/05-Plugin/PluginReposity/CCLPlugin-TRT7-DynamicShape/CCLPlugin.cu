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

#include "CCLPlugin.h"

__global__ void initializeLabelKernel(float *dPixelScore, float *dLinkScore, int *labelMap, int2 *linkMap, int batchSize, int nHeight, int nWidth, float minPixelScore, float minLinkScore)
{ // initialize the label and connect table
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nWidth + 2 || y >= nHeight + 2) // x and y used as the index of the big map
        return;

    const int idx = y * (nWidth + 2) + x;
    for (int i = 0; i < batchSize; i++, dPixelScore += nWidth * nHeight, labelMap += (nWidth + 2) * (nHeight + 2)) // root point will be labelMap[idx] == idx
        labelMap[idx] = (x == 0 || x == nWidth + 1 || y == 0 || y == nHeight + 1 || dPixelScore[(y - 1) * nWidth + (x - 1)] < minPixelScore) ? INT_MAX : idx;

    if (x >= nWidth || y >= nHeight) // x and y used as the index of the small map
        return;

    for (int i = 0; i < batchSize; i++, dLinkScore += nWidth * nHeight * 8, linkMap += (nWidth + 2) * (nHeight + 2)) // compress 8 link of the point into char8
    {
        char8 linkChar8;
#pragma unroll
        for (int j = 0; j < 8; j++)
            linkChar8.c8[j] = dLinkScore[j * nWidth * nHeight + y * nWidth + x] > minLinkScore;
        linkMap[(nWidth + 2) * (y + 1) + (x + 1)] = linkChar8.i2;
    }
}

__device__ __inline__ int8 getNeighbor(int idx, int nWidth) // get the 8 neighbor of the center point
{
    return int8 {
        idx - (nWidth + 2) - 1, idx - (nWidth + 2), idx - (nWidth + 2) + 1, idx - 1, idx + 1, idx + (nWidth + 2) - 1, idx + (nWidth + 2), idx + (nWidth + 2) + 1};
}

__global__ void initializeConnectKernel(char8 *linkMap, int batchSize, int nHeight, int nWidth) // initialize the connect table
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y * blockDim.y + threadIdx.y, x = ix, y = x % 2 ? (iy * 2 + 1) : (iy * 2);
    if (x >= nWidth || y >= nHeight)
        return;

    const int idx      = (y + 1) * (nWidth + 2) + (x + 1); // x and y used as the index of the small map, masaic sampling
    int8      neighbor = getNeighbor(idx, nWidth);
    for (int i = 0; i < batchSize; i++, linkMap += (nWidth + 2) * (nHeight + 2))
    {
        char8 linkChar8 {((int2 *)linkMap)[idx]};
        for (int i = 0; i < 8; ++i)
        {
            linkChar8.c8[i] |= linkMap[neighbor[i]].c8[7 - i]; // synchronize with my 8 neighbor
            if (i == 1 || i == 6 || i == 3 || i == 4)
                linkMap[neighbor[i]].c8[7 - i] = linkChar8.c8[i]; // adjust my 4 edge-neighbor's edge link
        }
        ((int2 *)linkMap)[idx]     = linkChar8.i2;
        linkMap[neighbor[3]].c8[2] = linkMap[neighbor[1]].c8[5] |= linkMap[neighbor[3]].c8[2]; // synchronize my L neighbor's RU link and my U neighbor's LD link
        linkMap[neighbor[3]].c8[7] = linkMap[neighbor[6]].c8[0] |= linkMap[neighbor[3]].c8[7]; // synchronize my L neighbor's RD link and my D neighbor's LU link
    }
}

__device__ int findMinNeighbor(int idx, int *labelMap, int2 *linkMap, int nWidth, bool updateNeighbor)
{
    int8  neighbor = getNeighbor(idx, nWidth);
    char8 connected {linkMap[idx]};

    int neighborLabel[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        neighborLabel[i] = labelMap[neighbor[i]];

    int minNeighbor =
        min(
            min(
                min(connected.c8[0] ? neighborLabel[0] : INT_MAX, connected.c8[1] ? neighborLabel[1] : INT_MAX),
                min(connected.c8[2] ? neighborLabel[2] : INT_MAX, connected.c8[3] ? neighborLabel[3] : INT_MAX)),
            min(
                min(connected.c8[4] ? neighborLabel[4] : INT_MAX, connected.c8[5] ? neighborLabel[5] : INT_MAX),
                min(connected.c8[6] ? neighborLabel[6] : INT_MAX, connected.c8[7] ? neighborLabel[7] : INT_MAX)));

    if (!updateNeighbor)
        return minNeighbor;

    int minValue = min(minNeighbor, labelMap[idx]);
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        if (connected.c8[i] && neighborLabel[i] != INT_MAX && neighborLabel[i] > minValue)
            labelMap[neighborLabel[i]] = minValue;
    }
    return minNeighbor;
}

__global__ void linkKernel(int *labelMap, int2 *linkMap, int batchSize, int nHeight, int nWidth, int *needLink, bool bUpdateNeighbor)
{ // find new link between neighbor
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nWidth || y >= nHeight) // x and y used as the index of the small map
        return;

    const int idx = (y + 1) * (nWidth + 2) + (x + 1);
    for (int i = 0; i < batchSize; i++, labelMap += (nWidth + 2) * (nHeight + 2), linkMap += (nWidth + 2) * (nHeight + 2))
    {
        if (!needLink[i]) // only the picture of needLink[i] == 1 need to link
            continue;

        int label = labelMap[idx];
        if (label == INT_MAX)
            continue;

        int mn = findMinNeighbor(idx, labelMap, linkMap, nWidth, bUpdateNeighbor);
        if (mn < label) // root label of the point is always smaller than the point
            labelMap[idx] = mn;
    }
}

__global__ void ResolveKernel(int *labelMap, int2 *linkMap, int batchSize, int nHeight, int nWidth, int *needLink, int *pNeedNext)
{ // find root for every point
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nWidth || y >= nHeight) // x and y used as the index of the small map
        return;

    const int idx = (y + 1) * (nWidth + 2) + (x + 1);
    for (int i = 0; i < batchSize; i++, labelMap += (nWidth + 2) * (nHeight + 2), linkMap += (nWidth + 2) * (nHeight + 2))
    {
        if (!needLink[i]) // only the picture of needLink[i] == 1 need to resolve
            continue;

        int label = labelMap[idx];
        if (label == INT_MAX)
            continue;

        bool bNewLabel = false;
        while (labelMap[label] != label)
        {
            label     = labelMap[label];
            bNewLabel = true;
        }

        if (bNewLabel)
            labelMap[idx] = label;

        if (pNeedNext && findMinNeighbor(idx, labelMap, linkMap, nWidth, false) < label) // need more turn to resolve
            pNeedNext[i] = true;
    }
}

template<class T>
__global__ void copyValueKernel(T *pDst, T *pSrc)
{
    pDst[threadIdx.x] = pSrc[threadIdx.x];
}

__global__ void reduceKernel(int *pDst, int *pSrc)
{
    atomicAdd(pDst, pSrc[threadIdx.x]);
}

__global__ void setRootMinusOneKernel(int *labelMap, int batchSize, int nHeight, int nWidth)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nWidth || y >= nHeight) // x and y used as the index of the small map
        return;

    const int idx = (y + 1) * (nWidth + 2) + (x + 1);
    for (int i = 0; i < batchSize; i++, labelMap += (nWidth + 2) * (nHeight + 2))
    {
        int label = labelMap[idx];
        if (label == INT_MAX || label != idx)
            continue;
        labelMap[idx] = -1;
    }
}

__global__ void calculateAreaKernel(int *labelMap, int batchSize, int nHeight, int nWidth) // labelMap[root of a label] == -(area of the label)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nWidth || y >= nHeight)
        return;

    const int idx = (y + 1) * (nWidth + 2) + (x + 1);
    for (int i = 0; i < batchSize; i++, labelMap += (nWidth + 2) * (nHeight + 2))
    {
        int label = labelMap[idx];
        if (label == INT_MAX || label < 0)
            continue;

        atomicAdd(&labelMap[label], -1);
    }
}

__global__ void filterAndRemarkKernel(int *labelMap, int batchSize, int nHeight, int nWidth, int minArea, int *pnComponent)
{ // filter small component and replace each root with -ordinal
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nWidth || y >= nHeight)
        return;

    const int idx = (y + 1) * (nWidth + 2) + (x + 1);
    for (int i = 0; i < batchSize; i++, labelMap += (nWidth + 2) * (nHeight + 2))
    {
        int label = labelMap[idx];
        if (label >= 0) // not a root point
            continue;

        if (-labelMap[idx] < minArea) // the area of the label is too small, discard it
            labelMap[idx] = INT_MAX;
        else
            labelMap[idx] = -(atomicAdd(&pnComponent[i], 1) + 1); // give the root point a new label name as
    }
}

__global__ void broadcastLabelKernel(int *labelMap, int batchSize, int nHeight, int nWidth) // expand -ordinal to every pixel
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nWidth || y >= nHeight)
        return;

    const int idx = (y + 1) * (nWidth + 2) + (x + 1);
    for (int i = 0; i < batchSize; i++, labelMap += (nWidth + 2) * (nHeight + 2))
    {
        int label = labelMap[idx];
        if (label == INT_MAX || label < 0) // is a non-label point or root point
            continue;

        labelMap[idx] = labelMap[label]; // mark every not-root-point with the label of its root label
    }
}

__global__ void reverseLabelKernel(int *labelMap, int batchSize, int nHeight, int nWidth) // reverse the value of every labeled-point
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nWidth || y >= nHeight)
        return;

    const int idx = (y + 1) * (nWidth + 2) + (x + 1);
    for (int i = 0; i < batchSize; i++, labelMap += (nWidth + 2) * (nHeight + 2))
    {
        int label = labelMap[idx];
        if (label == INT_MAX)
            continue;

        labelMap[idx] = -label;
    }
}

class CCL
{
private:
    int   batchSize, nHeight, nWidth;
    int * labelMap       = NULL;
    int2 *linkMap        = NULL;
    int * whichNeedLink  = NULL;
    int * temp           = NULL;
    int * hNeedIteration = NULL;
    int * dNeedIteration = NULL;

public:
    CCL(void *workspace, int batchSize, int nHeight, const int nWidth):
        batchSize(batchSize), nHeight(nHeight), nWidth(nWidth)
    {
        //cudaMalloc(&labelMap,         ALIGNED(sizeof(int)     * batchSize * (nHeight+2) * (nWidth+2)  ));
        //cudaMalloc(&linkMap,          ALIGNED(sizeof(char8)   * batchSize * (nHeight+2) * (nWidth+2)  ));
        //cudaMalloc(&whichNeedLink,    ALIGNED(sizeof(int)     * batchSize                             ));
        //cudaMalloc(&temp,             ALIGNED(sizeof(int)     * batchSize                             ));
        //cudaHostAlloc(&hNeedIteration,  sizeof(int), cudaHostAllocMapped);
        //cudaHostGetDevicePointer(&dNeedIteration, hNeedIteration, 0);
        //cudaMemset(linkMap, 0, ALIGNED(sizeof(char8)   * batchSize * (nHeight+2) * (nWidth+2))); // the cudaMalloc version of the memory management

        const int sizeLabel       = ALIGNED(sizeof(int) * batchSize * (nHeight + 2) * (nWidth + 2));
        const int sizeConnect     = ALIGNED(sizeof(char8) * batchSize * (nHeight + 2) * (nWidth + 2));
        const int sizeNeedCurrent = ALIGNED(sizeof(int) * batchSize);
        const int sizeTemp        = ALIGNED(sizeof(int) * batchSize);
        labelMap                  = (int *)workspace;
        linkMap                   = (int2 *)((char *)workspace + sizeLabel);
        whichNeedLink             = (int *)((char *)workspace + sizeLabel + sizeConnect);
        temp                      = (int *)((char *)workspace + sizeLabel + sizeConnect + sizeNeedCurrent);
        cudaHostAlloc(&hNeedIteration, sizeof(int), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&dNeedIteration, hNeedIteration, 0);
        cudaMemset(linkMap, 0, sizeConnect);
    }

    ~CCL()
    {
        //cudaFree(labelMap);
        //cudaFree(linkMap);
        //cudaFree(whichNeedLink);
        //cudaFree(temp);
        //cudaFreeHost(hNeedIteration);
    }

    void calculate(const void *const *input, float minPixelScore, float minLinkScore, int minArea, cudaStream_t stream)
    {
        initializeLabelKernel<<<dim3((nWidth + 2 + 31) / 32, (nHeight + 2 + 31) / 32), dim3(32, 32), 0, stream>>>((float *)input[0], (float *)input[1], labelMap, linkMap, batchSize, nHeight, nWidth, minPixelScore, minLinkScore);
        initializeConnectKernel<<<dim3((nWidth + 31) / 32, ((nHeight + 1) / 2 + 31) / 32), dim3(32, 32), 0, stream>>>((char8 *)linkMap, batchSize, nHeight, nWidth);

        cudaMemsetAsync(whichNeedLink, 1, batchSize * sizeof(int), stream);
        for (int i = 0, needIteration = 1; needIteration; ++i)
        {
            linkKernel<<<dim3((nWidth + 31) / 32, (nHeight + 31) / 32), dim3(32, 32), 0, stream>>>(labelMap, linkMap, batchSize, nHeight, nWidth, whichNeedLink, i != 0);

            cudaMemsetAsync(temp, 0, sizeof(int) * batchSize, stream); // temp used as 'whichNeedLink' in the next turn

            ResolveKernel<<<dim3((nWidth + 31) / 32, (nHeight + 31) / 32), dim3(32, 32), 0, stream>>>(labelMap, linkMap, batchSize, nHeight, nWidth, whichNeedLink, i ? temp : NULL);

            if (i == 0)
                continue;

            copyValueKernel<<<1, batchSize, 0, stream>>>(whichNeedLink, temp); // replace the value of whichNeedLink

            cudaMemsetAsync(dNeedIteration, 0, sizeof(int), stream); // judge whether to iterate more turn
            reduceKernel<<<1, batchSize, 0, stream>>>(dNeedIteration, temp);
            cudaStreamSynchronize(stream);
            needIteration = *hNeedIteration;
        }

        cudaMemsetAsync(temp, 0, batchSize * sizeof(int), stream); // temp used as the count of connected component in each picture
        setRootMinusOneKernel<<<dim3((nWidth + 31) / 32, (nHeight + 31) / 32), dim3(32, 32), 0, stream>>>(labelMap, batchSize, nHeight, nWidth);
        calculateAreaKernel<<<dim3((nWidth + 31) / 32, (nHeight + 31) / 32), dim3(32, 32), 0, stream>>>(labelMap, batchSize, nHeight, nWidth);
        filterAndRemarkKernel<<<dim3((nWidth + 31) / 32, (nHeight + 31) / 32), dim3(32, 32), 0, stream>>>(labelMap, batchSize, nHeight, nWidth, minArea, temp);
        broadcastLabelKernel<<<dim3((nWidth + 31) / 32, (nHeight + 31) / 32), dim3(32, 32), 0, stream>>>(labelMap, batchSize, nHeight, nWidth);
        reverseLabelKernel<<<dim3((nWidth + 31) / 32, (nHeight + 31) / 32), dim3(32, 32), 0, stream>>>(labelMap, batchSize, nHeight, nWidth);
    }

    void exportResult(void *const *output, cudaStream_t stream)
    {
        cudaMemcpy2DAsync((void *)output[0], sizeof(int) * nWidth, (void *)(labelMap + nWidth + 3), sizeof(int) * (nWidth + 2), sizeof(int) * nWidth, nHeight, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync((void *)output[1], (void *)temp, sizeof(int) * batchSize, cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }
};

int CCLPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream)
{
    CCL ccl(workspace, inputDesc[0].dims.d[0], m.height, m.width);
    ccl.calculate(inputs, m.minPixelScore, m.minLinkScore, m.minArea, stream);
    ccl.exportResult(outputs, stream);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(CCLPluginCreator);
