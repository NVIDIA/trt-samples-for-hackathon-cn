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

#include "OnehotPlugin.h"
#include <thread>
#include <stdio.h>
#include <nvfunctional>
#include <chrono>

using namespace nvinfer1;
using namespace plugin;

__global__ void OnehotPluginKernel(int *pArgmax, float *pOnehotPluginEncode, int depth)
{
    int batch_id = blockIdx.x;
    int elem_id = blockIdx.y * blockDim.x + threadIdx.x;

    // Warning: assume [off_value, on_value] is [0, 1]
    float value = .0f;  // initialize the value to zero

    if (elem_id < depth)
    {
        if (pArgmax[batch_id] == elem_id)
            value = 1.0f;
        pOnehotPluginEncode[batch_id*depth + elem_id] = value;
    }
}

PluginFieldCollection OnehotPluginCreator::mFC{};
std::vector<PluginField> OnehotPluginCreator::mPluginAttributes;

int OnehotPlugin::enqueue(int batchsize, const void * const *inputs, void * const *outputs, void* workspace, cudaStream_t stream) noexcept {
    dim3 dimBlock; 
    dim3 dimGrid;

    if(m.depth > 1024){
    	dimBlock.x = 1024;   
    	dimGrid.y  = (m.depth + dimBlock.x - 1)/dimBlock.x;
    }
    else{
        dimBlock.x = m.depth;
        dimGrid.y = 1;
    }
    dimGrid.x  = batchsize * m.nRow;
    //std::cout << "dimBlock" << dimBlock.x << ", dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;

    OnehotPluginKernel<<<dimGrid, dimBlock, 0, stream>>>((int*)inputs[0],
                                                          (float*)outputs[0],
                                                          m.depth
                                                          );
    return 0;
}

REGISTER_TENSORRT_PLUGIN(OnehotPluginCreator);