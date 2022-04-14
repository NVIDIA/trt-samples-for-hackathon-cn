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

#include "AddPluginDyn.h"
#include "cuda_fp16.h"
#include <chrono>
#include <thread>

template<typename T>
__global__ void AddValue(T *pDst, T *pSrc, int n, T valueToAdd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("[kernel]%f,%f,%f\n",float(pSrc[0]),float(valueToAdd),float(pSrc[0] + valueToAdd));
    }
    //pDst[x    ] = pSrc[x];
    //pDst[x + n] = pSrc[x] + valueToAdd;
    pDst[x] = pSrc[x] + valueToAdd;
}

template<typename T>
__global__ void gpuPrint(T * p, int n)
{
    printf("[gpuPrint]\n");
    for(int i = 0;i < n;i++)
        printf("%7.4f,",float(p[i]));
    printf("\n");
}

int AddPluginDyn::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
    int n = 1;
    printf("[enqueue]\n");
    for (int i = 0; i < inputDesc->dims.nbDims; i++) {
        printf("%d,",inputDesc->dims.d[i]);
        n *= inputDesc->dims.d[i];
    }
    
    printf("\nn=%d,valueToAdd=%f,scale=%f,v/s=%d\n",n,m.valueToAdd,m.scale,int((int8_t)(m.valueToAdd / m.scale)));
    
    if (m.dataType == nvinfer1::DataType::kFLOAT) {
        std::cout << "Running fp32 kernel" << std::endl;
        std::this_thread::sleep_for(20ms);
        
        (gpuPrint<float>)<<<1,1,0,stream>>>((float*)inputs[0],10);
        
        AddValue<<<(n + 1023) / 1024, 1024>>>((float *)outputs[0], (float *)inputs[0], n, m.valueToAdd);
    } else if (m.dataType == nvinfer1::DataType::kHALF) {
        std::cout << "Running fp16 kernel" << std::endl;
        std::this_thread::sleep_for(10ms);
        
        (gpuPrint<__half>)<<<1,1,0,stream>>>((__half *)inputs[0],10);
                
        AddValue<<<(n + 1023) / 1024, 1024>>>((__half *)outputs[0], (__half *)inputs[0], n, (__half)m.valueToAdd);
    } else {
        std::cout << "Running int8 kernel" << std::endl;
        std::this_thread::sleep_for(0ms);
        
        (gpuPrint<int8_t>)<<<1,1,0,stream>>>((int8_t *)inputs[0],10);
        cudaStreamSynchronize(stream);
        printf("\n");
                
        AddValue<<<(n + 1023) / 1024, 1024,0,stream>>>((int8_t *)outputs[0], (int8_t *)inputs[0], n, (int8_t)(m.valueToAdd / m.scale));
        
        (gpuPrint<int8_t>)<<<1,1,0,stream>>>((int8_t *)inputs[0],10);
        cudaStreamSynchronize(stream);
        printf("\n");
                
        (gpuPrint<int8_t>)<<<1,1,0,stream>>>(((int8_t *)outputs[0]),10);
        cudaStreamSynchronize(stream);
        printf("\n");
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(AddPluginDynCreator);
