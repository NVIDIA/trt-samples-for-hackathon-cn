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

/*// 单 thread 多数据，针对 128 线程 256 数据，废弃
template <typename T>
__global__ void scanAxis0Large(const T * input, T * output, int nWidth)
{
    const int bx = blockIdx.x, tx = threadIdx.x;

    //extern __shared__ T row[]; // compile error, need some trick
    extern __shared__ __align__(sizeof(T)) unsigned char byte[];
    T *list = reinterpret_cast<T *>(byte);

    list[tx] = input[bx * nWidth + tx];
    list[blockDim.x + tx] = input[bx * nWidth + blockDim.x + tx];
    __syncthreads();

    typedef cub::BlockScan<T, 128> BlockScan;
    __shared__ typename BlockScan::TempStorage tempScan;
    //T &tDataScan = list[tx*2];
    //BlockScan(tempScan).InclusiveSum(tDataScan, tDataScan);
    T tDataScan[2];
    tDataScan[0] = list[tx*2];
    tDataScan[1] = list[tx*2+1];
    BlockScan(tempScan).InclusiveSum(tDataScan, tDataScan);
    __syncthreads();

    output[bx * nWidth + 2*tx] = tDataScan[0];
    output[bx * nWidth + 2*tx+1] = tDataScan[1];
}
*/

template <typename T>
__global__ void scanAxis1Small(const T * input, T * output, const int nHeight, const int nWidth)
{
    const int bx = blockIdx.x, tx = threadIdx.x, index = bx * nWidth * nHeight + tx;

    if(tx >= nWidth)
        return;

    T sum = T(0);
    for(int i = 0; i < nWidth * nHeight; i += nWidth)
    {
        sum += input[index + i];
        output[index + i] = sum;
    }
}

template <typename T>
__global__ void scanAxis2Small(const T * input, T * output, const int nChannel, const int nWidth)
{
    const int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, index = by * nChannel * gridDim.x * nWidth + bx * nWidth + tx;

    if(tx >= nWidth)
        return;

    T sum = T(0);
    for(int i = 0; i < nChannel * gridDim.x * nWidth; i += gridDim.x * nWidth)
    {
        sum += input[index + i];
        output[index + i] = sum;
    }
}

template <typename T>
__global__ void scanAxis3Small(const T * input, T * output, const int nN, const int nWidth)
{
    const int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, index = by * gridDim.x * nWidth + bx * nWidth + tx;

    if(tx >= nWidth)
        return;

    T sum = T(0);
    for(int i = 0; i < nN * gridDim.y * gridDim.x * nWidth; i += gridDim.y * gridDim.x * nWidth)
    {
        sum += input[index + i];
        output[index + i] = sum;
    }
}

int CumSumPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    const int condition = (m.nDim - m.axis - 1) * 8 + m.kernelKind * 4 + m.datatype;
#if DEBUG
    printf("nDim=%d,axis=%d,datatype=%d,n=%d,c=%d,h=%d,w=%d,kernelKind=%d,condition=%d\n",m.nDim,m.axis,m.datatype,m.n,m.c,m.h,m.w,m.kernelKind,condition);
#endif
    switch( condition )
    {
    // w 轴（最低轴）
    case 0:     // small kernel, float32
        (scanAxis0Small<float>)     <<< m.n * m.c * m.h, 32, sizeof(float)*32, stream>>>    ((float*)inputs[0], (float*)outputs[0], m.w);
        break;
    case 1:     // small kernel, float16
        (scanAxis0Small<__half>)    <<< m.n * m.c * m.h, 32, sizeof(__half)*32, stream>>>   ((__half*)inputs[0], (__half*)outputs[0], m.w);
        break;
    //case 2:   // int8
    case 3:
        (scanAxis0Small<int>)       <<< m.n * m.c * m.h, 32, sizeof(int)*32, stream>>>      ((int*)inputs[0], (int*)outputs[0], m.w);
        break;  // small kernel, int32

    case 4:
        (scanAxis0Large<float>)     <<< m.n * m.c * m.h, 1024, sizeof(float)*1024, stream>>>  ((float*)inputs[0], (float*)outputs[0], m.w);
        break;  // large kernel, float32
    case 5:
        (scanAxis0Large<__half>)    <<< m.n * m.c * m.h, 1024, sizeof(__half)*1024, stream>>> ((__half*)inputs[0], (__half*)outputs[0], m.w);
        break;  // large kernel, float16
    //case 6:
    case 7:
        (scanAxis0Large<int>)       <<< m.n * m.c * m.h, 1024, sizeof(int)*1024, stream>>> ((int*)inputs[0], (int*)outputs[0], m.w);
        break;  // large kernel, int32

    // h 轴（次低轴）
    case 8:
    case 12:
        (scanAxis1Small<float>)     <<< m.n * m.c, ALIGN32(m.w), 0, stream>>> ((float*)inputs[0], (float*)outputs[0], m.h, m.w);
        break;  // large kernel, float32
    case 9:
    case 13:
        (scanAxis1Small<__half>)    <<< m.n * m.c, ALIGN32(m.w), 0, stream>>> ((__half*)inputs[0], (__half*)outputs[0], m.h, m.w);
        break;  // large kernel, float16
    //case 10:
    case 11:
    case 15:
        (scanAxis1Small<int>)       <<< m.n * m.c, ALIGN32(m.w), 0, stream>>> ((int*)inputs[0], (int*)outputs[0], m.h, m.w);
        break;  // large kernel, int32

    // c 轴（次高轴）
    case 16:
    case 20:
        (scanAxis2Small<float>)     <<< dim3(m.h,m.n), ALIGN32(m.w), 0, stream>>> ((float*)inputs[0], (float*)outputs[0], m.c, m.w);
        break;  // large kernel, float32

    case 17:
    case 21:
        (scanAxis2Small<__half>)    <<< dim3(m.h,m.n), ALIGN32(m.w), 0, stream>>> ((__half*)inputs[0], (__half*)outputs[0], m.c, m.w);
        break;  // large kernel, float32
    case 19:
    case 23:
        (scanAxis2Small<int>)       <<< dim3(m.h,m.n), ALIGN32(m.w), 0, stream>>> ((int*)inputs[0], (int*)outputs[0], m.c, m.w);
        break;  // large kernel, float32

    // n 轴（最高轴）
    case 24:
    case 28:
        (scanAxis3Small<float>)     <<< dim3(m.h,m.c), ALIGN32(m.w), 0, stream>>> ((float*)inputs[0], (float*)outputs[0], m.n, m.w);
        break;  // large kernel, float32
    case 25:
    case 29:
        (scanAxis3Small<__half>)    <<< dim3(m.h,m.c), ALIGN32(m.w), 0, stream>>> ((__half*)inputs[0], (__half*)outputs[0], m.n, m.w);
        break;  // large kernel, float32
    case 27:
    case 31:
        (scanAxis3Small<int>)       <<< dim3(m.h,m.c), ALIGN32(m.w), 0, stream>>> ((int*)inputs[0], (int*)outputs[0], m.n, m.w);
        break;  // large kernel, float32
    default:
#if DEBUG
        printf("[CumSumPlugin::enqueue()]Error condition! %d\n", condition);
#endif
        break;
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(CumSumPluginCreator);

