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

#include "GatherND.h"
#include "cuda_fp16.h"
#include <thread>
#include <chrono>

using namespace nvinfer1;
using namespace plugin;


// Device code
__global__ void GatherNDKernel(int *indices,
    float *params,
    int index_rank,
    const int *params_shape,
    int index_size,
    float *updates,
    int vec_size,
    int tar_num
) 
{
    int batch_idx = blockIdx.y;
    int tar_idx = blockIdx.x;

    int tar_size = gridDim.x;

    // calculate the index part idx based on indice input
    int indices_base =  (blockIdx.y * gridDim.x + blockIdx.x) * index_rank;
    int index_idx = 0;
    for (int i=0; i<index_rank; ++i){
        index_idx = index_idx * params_shape[i] + indices[indices_base+i];
    }

    // calculate the source idx of params
    int params_idx_base = batch_idx * index_size * vec_size +
                index_idx * vec_size;

    // calculate the target idx of updates
    int updates_idx_base = batch_idx * tar_size * vec_size +
                    tar_idx * vec_size;

    int vec_idx, params_idx, updates_idx;
    for (int i=0; i<tar_num; ++i){
        vec_idx = threadIdx.x + blockDim.x * i;
        if (vec_idx>=vec_size){
            break;
        }
        params_idx = params_idx_base + vec_idx;
        updates_idx = updates_idx_base + vec_idx;

        updates[updates_idx] = params[params_idx];
    }
}

PluginFieldCollection GatherNDCreator::mFC{};
std::vector<PluginField> GatherNDCreator::mPluginAttributes;

int GatherND::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    dim3 dimBlock;
    dim3 dimGrid;

    int index_size = multiplyArray(inputDesc[0].dims.d+m.batch_dims, m.index_rank);
    int vec_size = multiplyArray(inputDesc[0].dims.d+m.batch_dims+m.index_rank,m.updatesDim.nbDims-inputDesc[1].dims.nbDims+1);
    int tar_size = multiplyArray(inputDesc[1].dims.d+m.batch_dims,inputDesc[1].dims.nbDims-m.batch_dims-1);
    int batch_dim_size = multiplyArray(inputDesc[0].dims.d, m.batch_dims);

    dimBlock.x = vec_size >= 1024 ? 1024 : vec_size;
    dimGrid.x = tar_size;
    dimGrid.y = batch_dim_size;

    // copy params shape to device to calculate gather position
    cudaMemcpyAsync((int*)workspace, inputDesc[0].dims.d, inputDesc[0].dims.nbDims * sizeof(int32_t), cudaMemcpyHostToDevice, stream);


    // invoke kernel
    GatherNDKernel<<<dimGrid, dimBlock, 0, stream>>>((int*)inputs[1],
                                                (float*)inputs[0],
                                                m.index_rank,
                                                (int*)workspace+m.batch_dims,
                                                index_size,
                                                (float*)outputs[0],
                                                vec_size,
                                                ceilf(vec_size/float(dimBlock.x)));
    return 0;
}

REGISTER_TENSORRT_PLUGIN(GatherNDCreator);