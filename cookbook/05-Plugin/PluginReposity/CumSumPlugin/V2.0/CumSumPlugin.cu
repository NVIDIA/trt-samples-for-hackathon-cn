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

#include "CumSumPlugin.h"

using namespace nvinfer1;
using namespace plugin;

PluginFieldCollection CumSumPluginCreator::mFC{};
std::vector<PluginField> CumSumPluginCreator::mPluginAttributes;

template <typename T>
__global__ void scanAxis0Small(const T * input, T * output, int nWidth)
{
    const int bx = blockIdx.x, tx = threadIdx.x;
    //extern __shared__ T list[]; // compile error, need some trick
    extern __shared__ __align__(sizeof(T)) unsigned char byte[];
    T *list = reinterpret_cast<T *>(byte);
    if(tx >= nWidth)
        return;

    list[tx] = input[bx * nWidth + tx];
    typedef cub::WarpScan<T, 32> WarpScan;
    __shared__ typename WarpScan::TempStorage tempScan;
    T &tDataScan = list[tx];
    WarpScan(tempScan).InclusiveSum(tDataScan, tDataScan);

    output[bx * nWidth + tx] = list[tx];
}

template <typename T>
__global__ void scanAxis0Large(const T * input, T * output, int nWidth)
{
    const int bx = blockIdx.x, tx = threadIdx.x;

    //extern __shared__ T row[]; // compile error, need some trick
    extern __shared__ __align__(sizeof(T)) unsigned char byte[];
    T *list = reinterpret_cast<T *>(byte);
    if(tx >= nWidth)
        return;

    list[tx] = input[bx * nWidth + tx];
    typedef cub::BlockScan<T, 1024> BlockScan;
    __shared__ typename BlockScan::TempStorage tempScan;
    T &tDataScan = list[tx];
    BlockScan(tempScan).InclusiveSum(tDataScan, tDataScan);
    __syncthreads();

    output[bx * nWidth + tx] = list[tx];
}

template <typename T>
__global__ void scanAll(const T * input, T * output, const int nLoop, const int nWidth)
{
    const int tx    = threadIdx.x;
    const int index = blockIdx.y * gridDim.x * nLoop * nWidth + blockIdx.x * nWidth + tx;

    if(tx >= nWidth)
        return;

    T sum = T(0);
    for(int i = 0; i < nLoop * gridDim.x * nWidth; i += gridDim.x * nWidth)
    {
        sum += input[index + i];
        output[index + i] = sum;
    }
}

int CumSumPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    const int condition = (m.nDim - m.axis - 1) * 8 + m.kernelKind * 4 + m.datatype;
#if DEBUG
    printf("nDim=%d,axis=%d,datatype=%d,nHighDim=%d,nLowDim=%d,nLoop=%d,nWidth=%d,kernelKind=%d,condition=%d\n",m.nDim,m.axis,m.datatype,m.nHighDim,m.nLowDim,m.nLoop,m.nWidth,m.kernelKind,condition);
#endif
    switch( condition )
    {
    // w 轴（最低轴）
    case 0:     // small kernel, float32
        (scanAxis0Small<float>)     <<< m.nHighDim, 32, sizeof(float)*32, stream>>>    ((float*)inputs[0], (float*)outputs[0], m.nWidth);
        break;
    case 1:     // small kernel, float16
        (scanAxis0Small<__half>)    <<< m.nHighDim, 32, sizeof(__half)*32, stream>>>   ((__half*)inputs[0], (__half*)outputs[0], m.nWidth);
        break;
    //case 2:   // int8
    case 3:
        (scanAxis0Small<int>)       <<< m.nHighDim, 32, sizeof(int)*32, stream>>>      ((int*)inputs[0], (int*)outputs[0], m.nWidth);
        break;  // small kernel, int32

    case 4:
        (scanAxis0Large<float>)     <<< m.nHighDim, 1024, sizeof(float)*1024, stream>>>  ((float*)inputs[0], (float*)outputs[0], m.nWidth);
        break;  // large kernel, float32
    case 5:
        (scanAxis0Large<__half>)    <<< m.nHighDim, 1024, sizeof(__half)*1024, stream>>> ((__half*)inputs[0], (__half*)outputs[0], m.nWidth);
        break;  // large kernel, float16
    //case 6:
    case 7:
        (scanAxis0Large<int>)       <<< m.nHighDim, 1024, sizeof(int)*1024, stream>>> ((int*)inputs[0], (int*)outputs[0], m.nWidth);
        break;  // large kernel, int32

    case 8:
    case 12:
    case 16:
    case 20:
    case 24:
    case 28:
        (scanAll<float>)    <<< dim3(m.nLowDim,m.nHighDim), ALIGN32(m.nWidth), 0, stream>>> ((float*)inputs[0], (float*)outputs[0], m.nLoop, m.nWidth);
        break;
    case 9:
    case 13:
    case 17:
    case 21:
    case 25:
    case 29:
        (scanAll<__half>)   <<< dim3(m.nLowDim,m.nHighDim), ALIGN32(m.nWidth), 0, stream>>> ((__half*)inputs[0], (__half*)outputs[0], m.nLoop, m.nWidth);
        break;
    case 11:
    case 15:
    case 19:
    case 23:
    case 27:
    case 31:
        (scanAll<int>)      <<< dim3(m.nLowDim,m.nHighDim), ALIGN32(m.nWidth), 0, stream>>> ((int*)inputs[0], (int*)outputs[0], m.nLoop, m.nWidth);
        break;
    default:
#if DEBUG
        printf("[CumSumPlugin::enqueue()]Error condition! %d\n", condition);
#endif
        break;
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(CumSumPluginCreator);

