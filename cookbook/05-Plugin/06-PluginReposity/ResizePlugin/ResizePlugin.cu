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

#include "ResizePlugin.h"

__global__ void bilinearResize(float *pSrc, int nSrcPitch, int nW1, int nH1, int nC, float *pDst, int nW2, int nH2)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nW2 || y >= nH2)
        return;

    float alpha = min(max( (x+0.5f) * nW1  / nW2 -  0.5f, 0.0f ), nW1 - 1.0f);
    float beta  = min(max( (y+0.5f) * nH1 / nH2 - 0.5f, 0.0f ), nH1 - 1.0f);

    int srcL = (int)alpha, srcR = min(srcL + 1, nW1 - 1), srcU = (int)beta, srcD = min(srcU + 1, nH1 - 1);

    alpha = alpha - (int)alpha;
    beta  = beta  - (int)beta;
    for(int i=0;i<nC;i++)
    {
        float v1 = *(float*)((uint8_t*)pSrc + i * nSrcPitch * nH1 + srcU * nSrcPitch + srcL * sizeof(float));
        float v2 = *(float*)((uint8_t*)pSrc + i * nSrcPitch * nH1 + srcU * nSrcPitch + srcR * sizeof(float));
        float v3 = *(float*)((uint8_t*)pSrc + i * nSrcPitch * nH1 + srcD * nSrcPitch + srcL * sizeof(float));
        float v4 = *(float*)((uint8_t*)pSrc + i * nSrcPitch * nH1 + srcD * nSrcPitch + srcR * sizeof(float));
        *(float*)( (uint8_t*)pDst + (i*nW2*nH2+y*nW2+x)*sizeof(float) ) = v1*(1-alpha)*(1-beta) + v2*alpha*(1-beta) + v3*(1-alpha)*beta + v4*alpha*beta;
    }
}

int ResizePlugin::enqueue(int batchSize, const void * const *input, void **output, void* workspace, cudaStream_t stream)
{
    bilinearResize<<<dim3((m.w2 + 15)/16, (m.h2 + 15)/16), dim3(16, 16)>>>((float*)input[0], m.w1*sizeof(float), m.w1, m.h1, m.c, (float*)output[0], m.w2, m.h2);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(ResizePluginCreator);

