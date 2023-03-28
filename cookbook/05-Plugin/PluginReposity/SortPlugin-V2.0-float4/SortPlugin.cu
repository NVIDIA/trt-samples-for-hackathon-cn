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

#include "SortPlugin.h"

using namespace cub;

int SortPlugin::enqueue(int nBatch, const void *const *input, void **output, void *workspace, cudaStream_t stream)
{
    DoubleBuffer<float>  dKey;
    DoubleBuffer<float4> dValue;

    int iKey   = dKey.selector;
    int iValue = dValue.selector;

    dKey.d_buffers[iKey]         = (float *)input[0];
    dKey.d_buffers[1 - iKey]     = (float *)output[0];
    dValue.d_buffers[iValue]     = (float4 *)input[1];
    dValue.d_buffers[1 - iValue] = (float4 *)output[1];

    if (m.descending)
        DeviceRadixSort::SortPairsDescending(workspace, m.tempSpaceSize, dKey, dValue, m.nElement);
    else
        DeviceRadixSort::SortPairs(workspace, m.tempSpaceSize, dKey, dValue, m.nElement);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(SortPluginCreator);
