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

#include "WherePlugin.h"

__global__ void where32(const int *const condition, const float *const inputX, const float *const inputY, const int nTotal, float *const output)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nTotal)
        return;
    output[idx] = condition[idx] ? inputX[idx] : inputY[idx];
}

int WherePlugin::enqueue(int batchSize, const void *const *input, void **output, void *workspace, cudaStream_t stream)
{
    int nTotal = batchSize * m.nElement;
    where32<<<CEIL(nTotal, 32), 32, 0, stream>>>((int *)input[0], (float *)input[1], (float *)input[2], nTotal, (float *)output[0]);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(WherePluginCreator);
