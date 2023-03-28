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

#include "CumSumPlugin.h"

// 用于计算的 kernel
template<typename T, bool bInclusive>
__global__ void sumLastDimensionInWarp(const T *input, T *output, int nWidth)
{
    const int index = blockIdx.x * nWidth + threadIdx.x;
    //extern __shared__ T list[]; // compile error, need some trick
    extern __shared__ __align__(sizeof(T)) unsigned char byte[];
    T *                                                  list = reinterpret_cast<T *>(byte);
    if (threadIdx.x >= nWidth)
        return;

    list[threadIdx.x] = input[index];
    typedef cub::WarpScan<T, 32>              WarpScan;
    __shared__ typename WarpScan::TempStorage tempScan;
    T &                                       tDataScan = list[threadIdx.x];

    if (bInclusive)
    {
        WarpScan(tempScan).InclusiveSum(tDataScan, tDataScan);
    }
    else
    {
        WarpScan(tempScan).ExclusiveSum(tDataScan, tDataScan);
    }
    output[index] = list[threadIdx.x];
}

template __global__ void sumLastDimensionInWarp<float, true>(const float *input, float *output, int nWidth);
template __global__ void sumLastDimensionInWarp<float, false>(const float *input, float *output, int nWidth);
template __global__ void sumLastDimensionInWarp<__half, true>(const __half *input, __half *output, int nWidth);
template __global__ void sumLastDimensionInWarp<__half, false>(const __half *input, __half *output, int nWidth);
template __global__ void sumLastDimensionInWarp<int, true>(const int *input, int *output, int nWidth);
template __global__ void sumLastDimensionInWarp<int, false>(const int *input, int *output, int nWidth);

template<typename T, bool bInclusive>
__global__ void sumLastDimensionInBlock(const T *input, T *output, int nWidth)
{
    const int index = blockIdx.x * nWidth + threadIdx.x;
    //extern __shared__ T row[]; // compile error, need some trick
    extern __shared__ __align__(sizeof(T)) unsigned char byte[];
    T *                                                  list = reinterpret_cast<T *>(byte);
    if (threadIdx.x >= nWidth)
        return;

    list[threadIdx.x] = input[index];
    typedef cub::BlockScan<T, 1024>            BlockScan;
    __shared__ typename BlockScan::TempStorage tempScan;
    T &                                        tDataScan = list[threadIdx.x];

    if (bInclusive)
    {
        BlockScan(tempScan).InclusiveSum(tDataScan, tDataScan);
    }
    else
    {
        BlockScan(tempScan).ExclusiveSum(tDataScan, tDataScan);
    }
    __syncthreads();

    output[index] = list[threadIdx.x];
}

template __global__ void sumLastDimensionInBlock<float, true>(const float *input, float *output, int nWidth);
template __global__ void sumLastDimensionInBlock<float, false>(const float *input, float *output, int nWidth);
template __global__ void sumLastDimensionInBlock<__half, true>(const __half *input, __half *output, int nWidth);
template __global__ void sumLastDimensionInBlock<__half, false>(const __half *input, __half *output, int nWidth);
template __global__ void sumLastDimensionInBlock<int, true>(const int *input, int *output, int nWidth);
template __global__ void sumLastDimensionInBlock<int, false>(const int *input, int *output, int nWidth);

template<typename T>
__global__ void inclusiveSumHigherDimension(const T *input, T *output, const int nLoop, const int nWidth)
{
    const int index = blockIdx.y * gridDim.x * nLoop * nWidth + blockIdx.x * nWidth + threadIdx.x;

    if (threadIdx.x >= nWidth)
        return;

    T sum = T(0);
    for (int i = 0; i < nLoop * gridDim.x * nWidth; i += gridDim.x * nWidth)
    {
        sum += input[index + i];
        output[index + i] = sum;
    }
}

template __global__ void inclusiveSumHigherDimension<float>(const float *input, float *output, const int nLoop, const int nWidth);
template __global__ void inclusiveSumHigherDimension<__half>(const __half *input, __half *output, const int nLoop, const int nWidth);
template __global__ void inclusiveSumHigherDimension<int>(const int *input, int *output, const int nLoop, const int nWidth);

template<typename T>
__global__ void exclusiveSumHigherDimension(const T *input, T *output, const int nLoop, const int nWidth)
{
    const int index = blockIdx.y * gridDim.x * nLoop * nWidth + blockIdx.x * nWidth + threadIdx.x;

    if (threadIdx.x >= nWidth)
        return;

    T sum         = T(0);
    output[index] = sum;
    for (int i = 0; i < (nLoop - 1) * gridDim.x * nWidth; i += gridDim.x * nWidth)
    {
        sum += input[index + i];
        output[index + i + gridDim.x * nWidth] = sum;
    }
}

template __global__ void exclusiveSumHigherDimension<float>(const float *input, float *output, const int nLoop, const int nWidth);
template __global__ void exclusiveSumHigherDimension<__half>(const __half *input, __half *output, const int nLoop, const int nWidth);
template __global__ void exclusiveSumHigherDimension<int>(const int *input, int *output, const int nLoop, const int nWidth);

namespace nvinfer1
{
// class CumSumPlugin
CumSumPlugin::CumSumPlugin(const std::string &name, int nAxis, int bInclusive):
    name_(name)
{
    WHERE_AM_I();
    m_.nAxis      = nAxis;
    m_.bInclusive = bInclusive;
}

CumSumPlugin::CumSumPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

CumSumPlugin::~CumSumPlugin()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *CumSumPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new CumSumPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t CumSumPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType CumSumPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

DimsExprs CumSumPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    return inputs[0];
}

bool CumSumPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    switch (pos)
    {
    case 0:
        return (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF || inOut[0].type == DataType::kINT32) && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
    default: // should NOT be here!
        return false;
    }
    return false;
}

void CumSumPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    m_.nDim     = in[0].desc.dims.nbDims;
    m_.nAxis    = m_.nAxis % m_.nDim;
    m_.nHighDim = 1;
    m_.nLowDim  = 1;
    m_.nLoop    = in[0].desc.dims.d[m_.nAxis];
    m_.nWidth   = in[0].desc.dims.d[m_.nDim - 1];

    for (int i = 0; i < m_.nAxis; ++i)
    {
        m_.nHighDim *= in[0].desc.dims.d[i];
    }
    if (m_.nDim - m_.nAxis != 1) // not calculate the last nAxis
    {
        for (int i = m_.nAxis + 1; i < m_.nDim - 1; ++i)
            m_.nLowDim *= in[0].desc.dims.d[i];
    }
    return;
}

size_t CumSumPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CumSumPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    const int condition = int(m_.nDim - m_.nAxis == 1) * 16 + int(m_.nWidth > 32) * 8 + int(m_.bInclusive) * 4 + int(inputDesc[0].type);
#if DEBUG
    printf("nDim=%d,nAxis=%d,datatype=%d,nHighDim=%d,nLowDim=%d,nLoop=%d,nWidth=%d,condition=%d\n", m_.nDim, m_.nAxis, int(inputDesc[0].type), m_.nHighDim, m_.nLowDim, m_.nLoop, m_.nWidth, condition);
#endif
    switch (condition)
    {
    case 0: // higher axis, warp kernel, exclusive sum, float32
    case 8: // higher axis, block kernel, exclusive sum, float32
        (exclusiveSumHigherDimension<float>)<<<dim3(m_.nLowDim, m_.nHighDim), ALIGN_TO(m_.nWidth, 32), 0, stream>>>((float *)inputs[0], (float *)outputs[0], m_.nLoop, m_.nWidth);
        break;
    case 1: // higher axis, warp kernel, exclusive sum, float16
    case 9: // higher axis, block kernel, exclusive sum, float16
        (exclusiveSumHigherDimension<__half>)<<<dim3(m_.nLowDim, m_.nHighDim), ALIGN_TO(m_.nWidth, 32), 0, stream>>>((__half *)inputs[0], (__half *)outputs[0], m_.nLoop, m_.nWidth);
        break;
    //case 2:  // higher axis, warp kernel, exclusive sum, int8
    //case 10:  // higher axis, block kernel, exclusive sum, int8
    case 3:  // higher axis, warp kernel, exclusive sum, int32
    case 11: // higher axis, block kernel, exclusive sum, int32
        (exclusiveSumHigherDimension<int>)<<<dim3(m_.nLowDim, m_.nHighDim), ALIGN_TO(m_.nWidth, 32), 0, stream>>>((int *)inputs[0], (int *)outputs[0], m_.nLoop, m_.nWidth);
        break;
    case 4:  // higher axis, warp kernel, inclusive sum, float32
    case 12: // higher axis, block kernel, inclusive sum, float32
        (inclusiveSumHigherDimension<float>)<<<dim3(m_.nLowDim, m_.nHighDim), ALIGN_TO(m_.nWidth, 32), 0, stream>>>((float *)inputs[0], (float *)outputs[0], m_.nLoop, m_.nWidth);
        break;
    case 5:  // higher axis, warp kernel, inclusive sum, float16
    case 13: // higher axis, block kernel, inclusive sum, float16
        (inclusiveSumHigherDimension<__half>)<<<dim3(m_.nLowDim, m_.nHighDim), ALIGN_TO(m_.nWidth, 32), 0, stream>>>((__half *)inputs[0], (__half *)outputs[0], m_.nLoop, m_.nWidth);
        break;
    //case 6:  // higher axis, warp kernel, inclusive sum, int8
    //case 14:  // higher axis, block kernel, inclusive sum, int8
    case 7:  // higher axis, warp kernel, inclusive sum, int32
    case 15: // higher axis, block kernel, inclusive sum, int32
        (inclusiveSumHigherDimension<int>)<<<dim3(m_.nLowDim, m_.nHighDim), ALIGN_TO(m_.nWidth, 32), 0, stream>>>((int *)inputs[0], (int *)outputs[0], m_.nLoop, m_.nWidth);
        break;

    case 16: // last axis, warp kernel, exclusive sum, float32
        (sumLastDimensionInWarp<float, false>)<<<m_.nHighDim, 32, sizeof(float) * 32, stream>>>((float *)inputs[0], (float *)outputs[0], m_.nWidth);
        break;
    case 17: // last axis, warp kernel, exclusive sum, float16
        (sumLastDimensionInWarp<__half, false>)<<<m_.nHighDim, 32, sizeof(__half) * 32, stream>>>((__half *)inputs[0], (__half *)outputs[0], m_.nWidth);
        break;
    //case 18:  // last axis, warp kernel, exclusive sum, int8
    case 19: // last axis, warp kernel, exclusive sum, int32
        (sumLastDimensionInWarp<int, false>)<<<m_.nHighDim, 32, sizeof(int) * 32, stream>>>((int *)inputs[0], (int *)outputs[0], m_.nWidth);
        break;
    case 20: // last axis, warp kernel, inclusive sum, float32
        (sumLastDimensionInWarp<float, true>)<<<m_.nHighDim, 32, sizeof(float) * 32, stream>>>((float *)inputs[0], (float *)outputs[0], m_.nWidth);
        break;
    case 21: // last axis, warp kernel, inclusive sum, float16
        (sumLastDimensionInWarp<__half, true>)<<<m_.nHighDim, 32, sizeof(__half) * 32, stream>>>((__half *)inputs[0], (__half *)outputs[0], m_.nWidth);
        break;
    //case 22:  // last axis, warp kernel, inclusive sum, int8
    case 23: // last axis, warp kernel, inclusive sum, int32
        (sumLastDimensionInWarp<int, true>)<<<m_.nHighDim, 32, sizeof(int) * 32, stream>>>((int *)inputs[0], (int *)outputs[0], m_.nWidth);
        break;

    case 24: // last axis, block kernel, exclusive sum, float32
        (sumLastDimensionInBlock<float, false>)<<<m_.nHighDim, 1024, sizeof(float) * 1024, stream>>>((float *)inputs[0], (float *)outputs[0], m_.nWidth);
        break;
    case 25: // last axis, block kernel, exclusive sum, float16
        (sumLastDimensionInBlock<__half, false>)<<<m_.nHighDim, 1024, sizeof(__half) * 1024, stream>>>((__half *)inputs[0], (__half *)outputs[0], m_.nWidth);
        break;
    //case 26:  // last axis, block kernel, exclusive sum, int8
    case 27: // last axis, block kernel, exclusive sum, int1024
        (sumLastDimensionInBlock<int, false>)<<<m_.nHighDim, 1024, sizeof(int) * 1024, stream>>>((int *)inputs[0], (int *)outputs[0], m_.nWidth);
        break;
    case 28: // last axis, block kernel, inclusive sum, float32
        (sumLastDimensionInBlock<float, true>)<<<m_.nHighDim, 1024, sizeof(float) * 1024, stream>>>((float *)inputs[0], (float *)outputs[0], m_.nWidth);
        break;
    case 29: // last axis, block kernel, inclusive sum, float16
        (sumLastDimensionInBlock<__half, true>)<<<m_.nHighDim, 1024, sizeof(__half) * 1024, stream>>>((__half *)inputs[0], (__half *)outputs[0], m_.nWidth);
        break;
    //case 30:  // last axis, block kernel, inclusive sum, int8
    case 31: // last axis, block kernel, inclusive sum, int32
        (sumLastDimensionInBlock<int, true>)<<<m_.nHighDim, 1024, sizeof(int) * 1024, stream>>>((int *)inputs[0], (int *)outputs[0], m_.nWidth);
        break;

    default: // should NOT be here!
        printf("[CumSumPlugin::enqueue]Error condition! %d\n", condition);
    }
    return 0;
}

void CumSumPlugin::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t CumSumPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void CumSumPlugin::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t CumSumPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void CumSumPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void CumSumPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *CumSumPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *CumSumPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *CumSumPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

void CumSumPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void CumSumPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class CumSumPluginCreator
PluginFieldCollection    CumSumPluginCreator::fc_ {};
std::vector<PluginField> CumSumPluginCreator::attr_;

CumSumPluginCreator::CumSumPluginCreator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("bInclusive", nullptr, PluginFieldType::kINT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

CumSumPluginCreator::~CumSumPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2 *CumSumPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    int axis = -1, bInclusive = 1;

    std::map<std::string, int *> parameterMap {{"axis", &axis}, {"bInclusive", &bInclusive}};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
    }
    CumSumPlugin *pObj = new CumSumPlugin(name, axis, bInclusive);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2 *CumSumPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    CumSumPlugin *pObj = new CumSumPlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void CumSumPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *CumSumPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *CumSumPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *CumSumPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

const PluginFieldCollection *CumSumPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(CumSumPluginCreator);

} // namespace nvinfer1
