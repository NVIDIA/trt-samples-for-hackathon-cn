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

#include "SignPlugin.h"

__global__ void sign(const float *const input, const int nElement, float *const output)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nElement)
        output[idx] = copysignf(1.0f, input[idx]);
}

int SignPlugin::enqueue(int batchSize, const void *const *input, void **output, void *workspace, cudaStream_t stream)
{
    //printf(">%d,%d\n", batchSize, m.nElement);
    int nTotal = batchSize * m.nElement;
    sign<<<CEIL(nTotal, 128), 128, 0, stream>>>((float *)input[0], nTotal, (float *)output[0]);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(SignPluginCreator);
