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

template<typename T, typename OP_T, int TPB>
__global__ void LayerNormKernelSmall(const int nHiddenDimension, const T *input, const T *gamma, const T *beta, T *output)
{
    const int index       = blockIdx.x * nHiddenDimension + threadIdx.x;
    const T   denominator = T(1) / T(nHiddenDimension);
    OP_T      val         = 0;
    kvp<OP_T> threadData(0, 0);

    if (threadIdx.x < nHiddenDimension)
    {
        val       = input[index] * denominator;
        OP_T tmp0 = val * (OP_T)denominator, tmp1 = val * tmp0;
        threadData = mySum<OP_T>()(threadData, kvp<OP_T>(tmp0, tmp1));
    }

    using WarpReduce = cub::WarpReduce<kvp<OP_T>, TPB>;
    __shared__ typename WarpReduce::TempStorage temp;
    __shared__ OP_T                             mu, rsigma;

    const auto sumKV = WarpReduce(temp).Reduce(threadData, mySum<OP_T>()); // cub::Sum() 用不了？

    if (threadIdx.x == 0)
    {
        mu     = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + (OP_T)epsilon);
    }
    __syncthreads();

    if (threadIdx.x < nHiddenDimension)
    {
        const OP_T g = gamma[threadIdx.x], b = beta[threadIdx.x];
        output[index] = (val - mu) * rsigma * g + b;
    }
}

template __global__ void LayerNormKernelSmall<float, float, 32>(const int, const float *, const float *, const float *, float *);
template __global__ void LayerNormKernelSmall<__half, float, 32>(const int, const __half *, const __half *, const __half *, __half *);

template<typename T, typename OP_T, int TPB, int VPT>
__global__ void LayerNormKernelMedium(const int nHiddenDimension, const T *input, const T *gamma, const T *beta, T *output)
{
    // 考虑一个 block 上的寄存器使用量，当 nHiddenDimension 最大为 1024，元素为 float 时，
    // localX:      256 thread/block * 4 element/thread（即VPT） * 4 Byte/element = 4 KiB
    // localBeta:   1024 element / block * 4 Byte / element = 4 KiB
    // localGamma:  1024 element / block * 4 Byte / element = 4 KiB
    // localBias:   1024 element / block * 4 Byte / element = 4 KiB（这里没有）

    const int  index = blockIdx.x * nHiddenDimension + threadIdx.x * VPT;
    T          localX[VPT], localGamma[VPT], localBeta[VPT];
    const OP_T denominator = OP_T(1) / OP_T(nHiddenDimension);
    kvp<OP_T>  threadData(0, 0);

    copy<sizeof(T) * VPT>(&input[index], localX);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const OP_T tmp = (OP_T)localX[it] * denominator;
        threadData     = mySum<OP_T>()(threadData, kvp<OP_T>(tmp, tmp * (OP_T)localX[it]));
    }

    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], localBeta);
    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], localGamma);

    using BlockReduce = cub::BlockReduce<kvp<OP_T>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ OP_T                              mu, rsigma;

    const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, mySum<OP_T>());
    if (threadIdx.x == 0)
    {
        mu     = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + (OP_T)epsilon);
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        localX[it] = (OP_T)localGamma[it] * ((OP_T)localX[it] - mu) * rsigma + (OP_T)localBeta[it];
    }

    copy<sizeof(T) * VPT>(localX, &output[index]);
}

template __global__ void LayerNormKernelMedium<float, float, 64, 4>(const int, const float *, const float *, const float *, float *);
template __global__ void LayerNormKernelMedium<__half, float, 64, 4>(const int, const __half *, const __half *, const __half *, __half *);

template<typename T, typename OP_T, int TPB>
__global__ void LayerNormKernelLarge(const int nHiddenDimension, const T *input, const T *gamma, const T *beta, T *output)
{
    const int  offset      = blockIdx.x * nHiddenDimension;
    const OP_T denominator = OP_T(1) / OP_T(nHiddenDimension);
    kvp<OP_T>  threadData(0, 0);

    for (int i = threadIdx.x; i < nHiddenDimension; i += TPB)
    {
        const int  index = offset + i;
        OP_T       val   = input[index];
        const OP_T tmp   = val * denominator;
        threadData       = mySum<OP_T>()(threadData, kvp<OP_T>(tmp, tmp * val));
        output[index]    = val;
    }

    using BlockReduce = cub::BlockReduce<kvp<OP_T>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ OP_T                              mu, rsigma;

    const auto sumKV = BlockReduce(temp).Reduce(threadData, mySum<OP_T>());

    if (threadIdx.x == 0)
    {
        mu     = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + (OP_T)epsilon);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < nHiddenDimension; i += TPB)
    {
        const int index = offset + i;
        output[index]   = ((OP_T)output[index] - mu) * rsigma * (OP_T)gamma[i] + (OP_T)beta[i];
    }
}

template __global__ void LayerNormKernelLarge<float, float, 256>(const int, const float *, const float *, const float *, float *);
template __global__ void LayerNormKernelLarge<__half, float, 256>(const int, const __half *, const __half *, const __half *, __half *);

template<int TPB, int VPT>
__global__ void LayerNormKernelQDQ(const int nHiddenDimension, const int8_t *input, int8_t *output, const __half *gamma, const __half *beta, const float dqScaleIn, const float qScale)
{
    const int index = nHiddenDimension * blockIdx.x + threadIdx.x * VPT;
    int8_t    localX[VPT];
    __half    localXDQ[VPT], localBeta[VPT], localGamma[VPT];

    copy<sizeof(int8_t) * VPT>(&input[index], localX);
    __half2 loc = __floats2half2_rn(0.f, 0.f);

    const __half denominator = __half(1) / __half(nHiddenDimension);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const float tmp_in = localX[it];
        localXDQ[it]       = dqScaleIn * tmp_in;

        const __half  tmp  = localXDQ[it] * denominator;
        const __half2 tmp2 = __halves2half2(tmp, tmp * localXDQ[it]);
        loc                = loc + tmp2;
    }

    copy<sizeof(__half) * VPT>(&beta[threadIdx.x * VPT], localBeta);
    copy<sizeof(__half) * VPT>(&gamma[threadIdx.x * VPT], localGamma);

    using BlockReduce = cub::BlockReduce<__half2, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ __half                            mu;     // mean
    __shared__ __half                            rsigma; // 1 / std.dev.

    //const __half2 sum2 = BlockReduce(temp_storage).Reduce(loc, cub::Sum());
    const __half2 sum2 = BlockReduce(temp_storage).Reduce(loc, cub::Sum());

    if (threadIdx.x == 0)
    {
        mu     = __low2half(sum2);
        rsigma = rsqrt(__high2half(sum2) - mu * mu);
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const float tmp  = localGamma[it] * (localXDQ[it] - mu) * rsigma + localBeta[it];
        int         tmpq = __float2int_rn(qScale * tmp);
        tmpq             = max(-127, tmpq);
        tmpq             = min(127, tmpq);
        localX[it]       = tmpq;
    }

    copy<sizeof(int8_t) * VPT>(localX, &output[index]);
}

template __global__ void LayerNormKernelQDQ<32, 8>(const int, const int8_t *input, int8_t *output, const __half *gamma, const __half *beta, const float dqScaleIn, const float qScale);
template __global__ void LayerNormKernelQDQ<128, 8>(const int, const int8_t *input, int8_t *output, const __half *gamma, const __half *beta, const float dqScaleIn, const float qScale);

template<typename T>
int computeLayerNorm(const int gridSize, const int nHiddenDimension, const T *input, const T *gamma, const T *beta, T *output, cudaStream_t stream)
{
    constexpr int VPT = 16 / sizeof(T);
    if (nHiddenDimension <= 32)
    {
        constexpr int TPB = 32;
        (LayerNormKernelSmall<T, float, TPB>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, input, gamma, beta, output);
    }
    else if (nHiddenDimension == 256)
    {
        constexpr int TPB = 256 / VPT;
        (LayerNormKernelMedium<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, input, gamma, beta, output);
    }
    else if (nHiddenDimension == 1024)
    {
        constexpr int TPB = 1024 / VPT;
        (LayerNormKernelMedium<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, input, gamma, beta, output);
    }
    else
    {
        constexpr int TPB = 256;
        (LayerNormKernelLarge<T, float, TPB>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, input, gamma, beta, output);
    }
    CHECK(cudaPeekAtLastError());
    return 0;
}

template int computeLayerNorm<float>(const int, const int, const float *, const float *, const float *, float *, cudaStream_t);
template int computeLayerNorm<half>(const int, const int, const half *, const half *, const half *, half *, cudaStream_t);

int computeLayerNormDQQ(const int gridSize, const int nHiddenDimension, const int8_t *input, const __half *gamma, const __half *beta, int8_t *output, const float dqScaleIn, const float qScale, cudaStream_t stream)
{
    constexpr int VPT = 16 / sizeof(__half);
    if (nHiddenDimension == 256)
    {
        constexpr int TPB = 256 / VPT;
        (LayerNormKernelQDQ<TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, input, output, gamma, beta, dqScaleIn, qScale);
    }
    else if (nHiddenDimension == 1024)
    {
        constexpr int TPB = 1024 / VPT;
        (LayerNormKernelQDQ<TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, input, output, gamma, beta, dqScaleIn, qScale);
    }
    else
    {
        printf("[computeLayerNormDQQ] Unsupport hidden dimension %d!\n", nHiddenDimension);
        exit(0);
    }
    CHECK(cudaPeekAtLastError());
    return 0;
}

namespace nvinfer1
{
// class LayerNormPlugin
LayerNormPlugin::LayerNormPlugin(const std::string &name, const int nHiddenDimension):
    name_(name)
{
    WHERE_AM_I();
    m_.nHiddenDimension = nHiddenDimension;
}

LayerNormPlugin::LayerNormPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

LayerNormPlugin::~LayerNormPlugin()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *LayerNormPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new LayerNormPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t LayerNormPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType LayerNormPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

DimsExprs LayerNormPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    return inputs[0];
}

bool LayerNormPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    switch (pos)
    {
    case 0:
        return (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF) && inOut[0].format == TensorFormat::kLINEAR ||
               inOut[0].type == DataType::kINT8 && (inOut[0].format == TensorFormat::kCHW4 || inOut[0].format == TensorFormat::kCHW32);
    case 1:
    case 2:
        return inOut[pos].type == inOut[0].type || inOut[0].type == DataType::kINT8 && inOut[pos].type == DataType::kHALF;
    case 3:
        return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
    default: // should NOT be here!
        return false;
    }
    return false;
}

void LayerNormPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return;
}

size_t LayerNormPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    const int gridSize = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    int       status   = -1;

    switch (int(inputDesc[0].type))
    {
    case int(DataType::kFLOAT):
    {
#ifdef DEBUG
        printf("[enqueue]float32 path\n");
#endif
        const auto input  = static_cast<const float *>(inputs[0]);
        const auto gamma  = static_cast<const float *>(inputs[1]);
        const auto beta   = static_cast<const float *>(inputs[2]);
        auto       output = static_cast<float *>(outputs[0]);

        status = computeLayerNorm<float>(gridSize, m_.nHiddenDimension, input, gamma, beta, output, stream);
        break;
    }
    case int(DataType::kHALF):
    {
#ifdef DEBUG
        printf("[enqueue]float16 path\n");
#endif
        const auto input  = static_cast<const half *>(inputs[0]);
        const auto gamma  = static_cast<const half *>(inputs[1]);
        const auto beta   = static_cast<const half *>(inputs[2]);
        auto       output = static_cast<half *>(outputs[0]);

        status = computeLayerNorm<half>(gridSize, m_.nHiddenDimension, input, gamma, beta, output, stream);
        break;
    }
    case int(DataType::kINT8):
    {
#ifdef DEBUG
        printf("[enqueue]int8 path\n");
#endif
        const float dqScaleIn = inputDesc[0].scale;
        const float qScale    = 1.f / outputDesc[0].scale;
        const auto  input     = static_cast<const int8_t *>(inputs[0]);
        auto        output    = static_cast<int8_t *>(outputs[0]);
        const auto  gamma     = static_cast<const half *>(inputs[1]);
        const auto  beta      = static_cast<const half *>(inputs[2]);

        status = computeLayerNormDQQ(gridSize, m_.nHiddenDimension, input, gamma, beta, output, dqScaleIn, qScale, stream);
        break;
    }
    default:
    {
        printf("DataType not support!\n");
    }
    }
    return status;
}

void LayerNormPlugin::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t LayerNormPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void LayerNormPlugin::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t LayerNormPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void LayerNormPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void LayerNormPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *LayerNormPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *LayerNormPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *LayerNormPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

void LayerNormPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void LayerNormPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class LayerNormPluginCreator
PluginFieldCollection    LayerNormPluginCreator::fc_ {};
std::vector<PluginField> LayerNormPluginCreator::attr_;

LayerNormPluginCreator::LayerNormPluginCreator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("nHiddenDimension", nullptr, PluginFieldType::kINT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

LayerNormPluginCreator::~LayerNormPluginCreator()
{
    WHERE_AM_I();
}

const char *LayerNormPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *LayerNormPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

IPluginV2 *LayerNormPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    int                          nHiddenDimension = 0;
    std::map<std::string, int *> parameterMap {{"nHiddenDimension", &nHiddenDimension}};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
    }
    return new LayerNormPlugin(name, nHiddenDimension);
}

IPluginV2 *LayerNormPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new LayerNormPlugin(name, serialData, serialLength);
}

void LayerNormPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *LayerNormPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const PluginFieldCollection *LayerNormPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

} // namespace nvinfer1
