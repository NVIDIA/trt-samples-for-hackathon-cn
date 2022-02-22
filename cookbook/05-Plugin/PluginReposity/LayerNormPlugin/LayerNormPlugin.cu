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

#include "LayerNormPlugin.h"

using namespace nvinfer1;
using namespace plugin;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

template<typename T, int n>
__global__ void layerNormKernel(T *pInput, T *pGamma, T *pBeta, T *pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * n + tx;
    T _x = pInput[index], _b = pGamma[tx], _a = pBeta[tx];

    __shared__ T mean_shared, var_shared;

    typedef cub::BlockReduce<T, n> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    T& ref0 = _x;
    T sum = BlockReduce(temp).Sum(ref0);
    //__syncthreads();
    if(tx == 0)
        mean_shared = sum / T(n);
    __syncthreads();

    T moment = _x - mean_shared, moment2 = moment * moment;
    T& ref1 = moment2;
    T var = BlockReduce(temp).Sum(ref1);
    //__syncthreads();
    if(tx == 0)
        var_shared = var / T(n);
    __syncthreads();

    pOutput[index] = (moment) * (T)rsqrtf(var_shared + T(EPSILON)) * _b + _a;
}

template <typename T>
int32_t LayerNormPlugin<T>::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1], nValuePerBlock = inputDesc[0].dims.d[2];
    if (inputDesc[0].type == DataType::kFLOAT)
    {
        switch(nValuePerBlock)
        {
        case 320:
            (layerNormKernel<float,320>) <<<nBlock, nValuePerBlock, 0, stream>>>((float *)inputs[0], (float *)inputs[1], (float *)inputs[2], (float *)outputs[0]);
            break;
        case 560: 
            (layerNormKernel<float,560>) <<<nBlock, nValuePerBlock, 0, stream>>>((float *)inputs[0], (float *)inputs[1], (float *)inputs[2], (float *)outputs[0]);
            break;
        default:    // shoulf NOT be here
            printf("[LayerNormPlugin<T>::enqueue] nValuePerBlock is not in [320,560]\n");
            break;
        }
    }
    else
    {
        switch(nValuePerBlock)
        {
        case 320:
            (layerNormKernel<half,320>) <<<nBlock, nValuePerBlock, 0, stream>>>((half *)inputs[0], (half *)inputs[1], (half *)inputs[2], (half *)outputs[0]);
            break;
        case 560: 
            (layerNormKernel<half,560>) <<<nBlock, nValuePerBlock, 0, stream>>>((half *)inputs[0], (half *)inputs[1], (half *)inputs[2], (half *)outputs[0]);
            break;
        default:    // shoulf NOT be here
            printf("[LayerNormPlugin<T>::enqueue] nValuePerBlock is not in [320,560]\n");
            break;
        }
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

