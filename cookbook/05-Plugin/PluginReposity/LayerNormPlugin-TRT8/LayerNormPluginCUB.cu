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

#include "LayerNormPluginCUB.h"

// class LayerNormPluginV1
__global__ void layerNormKernelV1(float *pInput, float *pOutput, float epsilon)
{
    const int tx = threadIdx.x, index = blockIdx.x * 256 + threadIdx.x;

    __shared__ float temp[128];

    float value0 = pInput[index];
    float value1 = pInput[index + 128];
    temp[tx]     = value0 + value1;
    __syncthreads();

    for (int stride = 64; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    float mean = temp[0] / 256;
    __syncthreads();

    temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean);
    __syncthreads();

    for (int stride = 64; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    float var = temp[0] / 256;

    pOutput[index]       = (value0 - mean) * rsqrtf(var + epsilon);
    pOutput[index + 128] = (value1 - mean) * rsqrtf(var + epsilon);
}

// class LayerNormPluginV2
template<typename T, int n>
__global__ void layerNormKernelV2(T *pInput, T *pOutput, float epsilon)
{
    const int tx = threadIdx.x, index = blockIdx.x * n + threadIdx.x;

    T _x = pInput[index];

    __shared__ T mean_shared, var_shared;

    typedef cub::BlockReduce<T, n>               BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    T &                                          ref0 = _x;
    T                                            sum  = BlockReduce(temp).Sum(ref0);
    //__syncthreads();
    if (tx == 0)
        mean_shared = sum / (T)n;
    __syncthreads();

    T  moment = _x - mean_shared, moment2 = moment * moment;
    T &ref1 = moment2;
    T  var  = BlockReduce(temp).Sum(ref1);
    //__syncthreads();
    if (tx == 0)
        var_shared = var / (T)n;
    __syncthreads();

    pOutput[index] = moment * (T)rsqrtf(var_shared + (T)epsilon);
}

// class LayerNormPluginV3
template<int VPT>
struct BytesToType;

template<>
struct BytesToType<2>
{
    using type = uint16_t;
};
template<>
struct BytesToType<4>
{
    using type = uint32_t;
};
template<>
struct BytesToType<8>
{
    using type = uint64_t;
};
template<>
struct BytesToType<16>
{
    using type = float4;
};

template<int Bytes>
__device__ inline void copy(const void *local, void *data)
{
    using T = typename BytesToType<Bytes>::type;

    const T *in  = static_cast<const T *>(local);
    T *      out = static_cast<T *>(data);
    *out         = *in;
}

template<typename T>
using kvp = cub::KeyValuePair<T, T>;

template<typename T>
struct mySum
{
    __host__ __device__ __forceinline__ kvp<T> operator()(const kvp<T> &a, const kvp<T> &b) const
    {
        return kvp<T>(a.key + b.key, a.value + b.value);
    }
};

template<typename T, int TPB, int VPT>
__global__ void layerNormKernelV3(const T *input, const T *gamma, const T *beta, T *output)
{
    const int   idx = blockIdx.x * 256 + VPT * threadIdx.x;
    T           localX[VPT], localGamma[VPT], localBeta[VPT];
    kvp<float>  localFloat2 = {0.f, 0.f};
    const float denominator = float(1) / float(256);

    copy<sizeof(T) * VPT>(&input[idx], localX);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const float tmp = denominator * (float)localX[it];
        localFloat2.key += tmp;
        localFloat2.value += tmp * (float)localX[it];
    }

    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], localGamma);
    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], localBeta);

    using BlockReduce = cub::BlockReduce<kvp<float>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float                             mu;     // mean
    __shared__ float                             rsigma; // 1 / std.dev.

    const kvp<float> sumKV = BlockReduce(temp).Reduce(localFloat2, mySum<float>());

    if (threadIdx.x == 0)
    {
        mu     = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + 1e-6);
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        localX[it] = ((float)localX[it] - mu) * rsigma * (float)localGamma[it] + (float)localBeta[it];
    }

    copy<sizeof(T) * VPT>(localX, &output[idx]);
}

template __global__ void layerNormKernelV3<float, 64, 4>(const float *, const float *, const float *, float *);
template __global__ void layerNormKernelV3<half, 32, 8>(const half *, const half *, const half *, half *);

// class LayerNormPluginV4
template<typename T, typename OP_T, int TPB>
__global__ void LayerNormSmallKernelV4(const int nHiddenDimension, const T *input, const T *gamma, const T *beta, T *output, const float epsilon)
{
    const int index       = blockIdx.x * nHiddenDimension + threadIdx.x;
    const T   denominator = T(1) / T(nHiddenDimension);
    OP_T      val         = 0;
    kvp<OP_T> threadData(0, 0);

    if (threadIdx.x < nHiddenDimension)
    {
        val        = input[index] * denominator;
        OP_T tmp0  = val * (OP_T)denominator;
        OP_T tmp1  = val * tmp0;
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

template __global__ void LayerNormSmallKernelV4<float, float, 32>(const int, const float *, const float *, const float *, float *, const float);
template __global__ void LayerNormSmallKernelV4<__half, float, 32>(const int, const __half *, const __half *, const __half *, __half *, const float);

template<typename T, typename OP_T, int TPB, int VPT>
__global__ void LayerNormMediumKernelV4(const int nHiddenDimension, const T *input, const T *gamma, const T *beta, T *output, const float epsilon)
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

template __global__ void LayerNormMediumKernelV4<float, float, 64, 4>(const int, const float *, const float *, const float *, float *, const float);
template __global__ void LayerNormMediumKernelV4<__half, float, 64, 4>(const int, const __half *, const __half *, const __half *, __half *, const float);

template<typename T, typename OP_T, int TPB>
__global__ void LayerNormLargeKernelV4(const int nHiddenDimension, const T *input, const T *gamma, const T *beta, T *output, const float epsilon)
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

template __global__ void LayerNormLargeKernelV4<float, float, 256>(const int, const float *, const float *, const float *, float *, const float);
template __global__ void LayerNormLargeKernelV4<__half, float, 256>(const int, const __half *, const __half *, const __half *, __half *, const float);

template<int TPB, int VPT>
__global__ void LayerNormQDQKernelV4(const int nHiddenDimension, const int8_t *input, int8_t *output, const __half *gamma, const __half *beta, const float dqScaleIn, const float qScale, const float epsilon)
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
        rsigma = rsqrt(__high2half(sum2) - mu * mu + (__half)epsilon);
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

template __global__ void LayerNormQDQKernelV4<32, 8>(const int, const int8_t *, int8_t *, const __half *, const __half *, const float, const float, const float);
template __global__ void LayerNormQDQKernelV4<128, 8>(const int, const int8_t *, int8_t *, const __half *, const __half *, const float, const float, const float);

template<typename T>
int computeLayerNormV4(const int gridSize, const int nHiddenDimension, const T *input, const T *gamma, const T *beta, T *output, const float epsilon, cudaStream_t stream)
{
    constexpr int VPT = 16 / sizeof(T);
    if (nHiddenDimension <= 32)
    {
        constexpr int TPB = 32;
        (LayerNormSmallKernelV4<T, float, TPB>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, input, gamma, beta, output, epsilon);
    }
    else if (nHiddenDimension == 256)
    {
        constexpr int TPB = 256 / VPT;
        (LayerNormMediumKernelV4<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, input, gamma, beta, output, epsilon);
    }
    else if (nHiddenDimension == 1024)
    {
        constexpr int TPB = 1024 / VPT;
        (LayerNormMediumKernelV4<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, input, gamma, beta, output, epsilon);
    }
    else
    {
        constexpr int TPB = 256;
        (LayerNormLargeKernelV4<T, float, TPB>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, input, gamma, beta, output, epsilon);
    }
    CHECK(cudaPeekAtLastError());
    return 0;
}

template int computeLayerNormV4<float>(const int, const int, const float *, const float *, const float *, float *, const float, cudaStream_t);
template int computeLayerNormV4<half>(const int, const int, const half *, const half *, const half *, half *, const float, cudaStream_t);

int computeLayerNormQDQV4(const int gridSize, const int nHiddenDimension, const int8_t *input, const __half *gamma, const __half *beta, int8_t *output, const float dqScaleIn, const float qScale, const float epsilon, cudaStream_t stream)
{
    constexpr int VPT = 16 / sizeof(__half);
    if (nHiddenDimension == 256)
    {
        constexpr int TPB = 256 / VPT;
        (LayerNormQDQKernelV4<TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, input, output, gamma, beta, dqScaleIn, qScale, epsilon);
    }
    else if (nHiddenDimension == 1024)
    {
        constexpr int TPB = 1024 / VPT;
        (LayerNormQDQKernelV4<TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, input, output, gamma, beta, dqScaleIn, qScale, epsilon);
    }
    else
    {
        printf("[computeLayerNormQDQV4] Unsupport hidden dimension %d!\n", nHiddenDimension);
        exit(0);
    }
    CHECK(cudaPeekAtLastError());
    return 0;
}

namespace nvinfer1
{
// class LayerNormPluginV1
LayerNormPluginV1::LayerNormPluginV1(const std::string &name, float epsilon):
    name_(name)
{
    WHERE_AM_I();
    m_.epsilon = epsilon;
}

LayerNormPluginV1::LayerNormPluginV1(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

LayerNormPluginV1::~LayerNormPluginV1()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *LayerNormPluginV1::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new LayerNormPluginV1(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t LayerNormPluginV1::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType LayerNormPluginV1::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return DataType::kFLOAT;
}

DimsExprs LayerNormPluginV1::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    return inputs[0];
}

bool LayerNormPluginV1::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    switch (pos)
    {
    case 0:
        return inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
    default: // should NOT be here!
        return false;
    }
    return false;
}

void LayerNormPluginV1::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return;
}

size_t LayerNormPluginV1::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t LayerNormPluginV1::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1]; // 仅用于处理 nEmbedding 为 256 的情况

    layerNormKernelV1<<<nBlock, 128, 0, stream>>>((float *)inputs[0], (float *)outputs[0], m_.epsilon);
    return 0;
}

void LayerNormPluginV1::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t LayerNormPluginV1::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void LayerNormPluginV1::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t LayerNormPluginV1::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void LayerNormPluginV1::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void LayerNormPluginV1::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *LayerNormPluginV1::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *LayerNormPluginV1::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *LayerNormPluginV1::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION_V1;
}

void LayerNormPluginV1::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void LayerNormPluginV1::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class LayerNormPluginV1Creator
PluginFieldCollection    LayerNormPluginV1Creator::fc_ {};
std::vector<PluginField> LayerNormPluginV1Creator::attr_;

LayerNormPluginV1Creator::LayerNormPluginV1Creator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

LayerNormPluginV1Creator::~LayerNormPluginV1Creator()
{
    WHERE_AM_I();
}

IPluginV2 *LayerNormPluginV1Creator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    float                          epsilon = 1.0e-5f;
    std::map<std::string, float *> parameterMap {{"epsilon", &epsilon}};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const float *>(fc->fields[i].data);
        }
    }
    return new LayerNormPluginV1(name, epsilon);
}

IPluginV2 *LayerNormPluginV1Creator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new LayerNormPluginV1(name, serialData, serialLength);
}

void LayerNormPluginV1Creator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *LayerNormPluginV1Creator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *LayerNormPluginV1Creator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *LayerNormPluginV1Creator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION_V1;
}

const PluginFieldCollection *LayerNormPluginV1Creator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginV1Creator);

// class LayerNormPluginV2
LayerNormPluginV2::LayerNormPluginV2(const std::string &name, float epsilon):
    name_(name)
{
    WHERE_AM_I();
    m_.epsilon = epsilon;
}

LayerNormPluginV2::LayerNormPluginV2(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

LayerNormPluginV2::~LayerNormPluginV2()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *LayerNormPluginV2::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new LayerNormPluginV2(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t LayerNormPluginV2::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType LayerNormPluginV2::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

DimsExprs LayerNormPluginV2::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    return inputs[0];
}

bool LayerNormPluginV2::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    switch (pos)
    {
    case 0:
        return (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF) && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
    default: // should NOT be here!
        return false;
    }
    return false;
}

void LayerNormPluginV2::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return;
}

size_t LayerNormPluginV2::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t LayerNormPluginV2::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1], nValuePerBlock = inputDesc[0].dims.d[2];

    if (inputDesc[0].type == DataType::kFLOAT)
    {
        switch (nValuePerBlock)
        {
        case 256: // 仅用于处理 nEmbedding 为 256 的情况
            (layerNormKernelV2<float, 256>)<<<nBlock, nValuePerBlock, 0, stream>>>((float *)inputs[0], (float *)outputs[0], m_.epsilon);
            break;
        default: // should NOT be here!
            printf("[LayerNormPluginV2::enqueue] nValuePerBlock = %d is not supported\n", nValuePerBlock);
            break;
        }
    }
    else
    {
        switch (nValuePerBlock)
        {
        case 256: // 仅用于处理 nEmbedding 为 256 的情况
            (layerNormKernelV2<half, 256>)<<<nBlock, nValuePerBlock, 0, stream>>>((half *)inputs[0], (half *)outputs[0], m_.epsilon);
            break;
        default: // should NOT be here!
            printf("[LayerNormPluginV2::enqueue] nValuePerBlock = %d is not supported\n", nValuePerBlock);
            break;
        }
    }
    return 0;
}

void LayerNormPluginV2::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t LayerNormPluginV2::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void LayerNormPluginV2::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t LayerNormPluginV2::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void LayerNormPluginV2::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void LayerNormPluginV2::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *LayerNormPluginV2::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *LayerNormPluginV2::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *LayerNormPluginV2::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION_V2;
}

void LayerNormPluginV2::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void LayerNormPluginV2::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class LayerNormPluginV2Creator
PluginFieldCollection    LayerNormPluginV2Creator::fc_ {};
std::vector<PluginField> LayerNormPluginV2Creator::attr_;

LayerNormPluginV2Creator::LayerNormPluginV2Creator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

LayerNormPluginV2Creator::~LayerNormPluginV2Creator()
{
    WHERE_AM_I();
}

IPluginV2 *LayerNormPluginV2Creator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    float                          epsilon = 1.0e-5f;
    std::map<std::string, float *> parameterMap {{"epsilon", &epsilon}};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const float *>(fc->fields[i].data);
        }
    }
    return new LayerNormPluginV2(name, epsilon);
}

IPluginV2 *LayerNormPluginV2Creator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new LayerNormPluginV2(name, serialData, serialLength);
}

void LayerNormPluginV2Creator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *LayerNormPluginV2Creator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *LayerNormPluginV2Creator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *LayerNormPluginV2Creator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION_V2;
}

const PluginFieldCollection *LayerNormPluginV2Creator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginV2Creator);

// class LayerNormPluginV3
LayerNormPluginV3::LayerNormPluginV3(const std::string &name, float epsilon):
    name_(name)
{
    WHERE_AM_I();
    m_.epsilon = epsilon;
}

LayerNormPluginV3::LayerNormPluginV3(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

LayerNormPluginV3::~LayerNormPluginV3()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *LayerNormPluginV3::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new LayerNormPluginV3(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t LayerNormPluginV3::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType LayerNormPluginV3::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

DimsExprs LayerNormPluginV3::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    return inputs[0];
}

bool LayerNormPluginV3::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    switch (pos)
    {
    case 0:
        return (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF) && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
    case 2:
    case 3:
        return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
    default: // should NOT be here!
        return false;
    }
    return false;
}

void LayerNormPluginV3::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return;
}

size_t LayerNormPluginV3::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t LayerNormPluginV3::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    const int gridSize = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];

    if (inputDesc[0].type == DataType::kFLOAT)
    {
        constexpr int VPT = 16 / sizeof(float);
        constexpr int TPB = 256 / VPT;
        (layerNormKernelV3<float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>((const float *)inputs[0], (const float *)inputs[1], (const float *)inputs[2], (float *)outputs[0]);
    }
    else
    {
        constexpr int VPT = 16 / sizeof(half);
        constexpr int TPB = 256 / VPT;
        (layerNormKernelV3<half, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>((const half *)inputs[0], (const half *)inputs[1], (const half *)inputs[2], (half *)outputs[0]);
    }
    return 0;
}

void LayerNormPluginV3::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t LayerNormPluginV3::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void LayerNormPluginV3::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t LayerNormPluginV3::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void LayerNormPluginV3::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void LayerNormPluginV3::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *LayerNormPluginV3::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *LayerNormPluginV3::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *LayerNormPluginV3::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION_V3;
}

void LayerNormPluginV3::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void LayerNormPluginV3::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class LayerNormPluginV3Creator
PluginFieldCollection    LayerNormPluginV3Creator::fc_ {};
std::vector<PluginField> LayerNormPluginV3Creator::attr_;

LayerNormPluginV3Creator::LayerNormPluginV3Creator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

LayerNormPluginV3Creator::~LayerNormPluginV3Creator()
{
    WHERE_AM_I();
}

IPluginV2 *LayerNormPluginV3Creator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    float                          epsilon = 1.0e-5f;
    std::map<std::string, float *> parameterMap {{"epsilon", &epsilon}};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const float *>(fc->fields[i].data);
        }
    }
    return new LayerNormPluginV3(name, epsilon);
}

IPluginV2 *LayerNormPluginV3Creator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new LayerNormPluginV3(name, serialData, serialLength);
}

void LayerNormPluginV3Creator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *LayerNormPluginV3Creator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *LayerNormPluginV3Creator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *LayerNormPluginV3Creator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION_V3;
}

const PluginFieldCollection *LayerNormPluginV3Creator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginV3Creator);

// class LayerNormPluginV4
LayerNormPluginV4::LayerNormPluginV4(const std::string &name, float epsilon):
    name_(name)
{
    WHERE_AM_I();
    m_.epsilon = epsilon;
}

LayerNormPluginV4::LayerNormPluginV4(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

LayerNormPluginV4::~LayerNormPluginV4()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *LayerNormPluginV4::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new LayerNormPluginV4(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t LayerNormPluginV4::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType LayerNormPluginV4::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

DimsExprs LayerNormPluginV4::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    return inputs[0];
}

bool LayerNormPluginV4::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
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

void LayerNormPluginV4::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return;
}

size_t LayerNormPluginV4::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t LayerNormPluginV4::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    const int gridSize    = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    const int nHiddenSize = inputDesc[0].dims.d[2];
    int       status      = -1;

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

        status = computeLayerNormV4<float>(gridSize, nHiddenSize, input, gamma, beta, output, m_.epsilon, stream);
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

        status = computeLayerNormV4<half>(gridSize, nHiddenSize, input, gamma, beta, output, m_.epsilon, stream);
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

        status = computeLayerNormQDQV4(gridSize, nHiddenSize, input, gamma, beta, output, dqScaleIn, qScale, m_.epsilon, stream);
        break;
    }
    default:
    {
        printf("DataType not support!\n");
    }
    }
    return status;
}

void LayerNormPluginV4::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t LayerNormPluginV4::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void LayerNormPluginV4::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t LayerNormPluginV4::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void LayerNormPluginV4::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void LayerNormPluginV4::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *LayerNormPluginV4::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *LayerNormPluginV4::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *LayerNormPluginV4::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION_V4;
}

void LayerNormPluginV4::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void LayerNormPluginV4::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class LayerNormPluginV4Creator
PluginFieldCollection    LayerNormPluginV4Creator::fc_ {};
std::vector<PluginField> LayerNormPluginV4Creator::attr_;

LayerNormPluginV4Creator::LayerNormPluginV4Creator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

LayerNormPluginV4Creator::~LayerNormPluginV4Creator()
{
    WHERE_AM_I();
}

IPluginV2 *LayerNormPluginV4Creator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    float                          epsilon = 1.0e-5f;
    std::map<std::string, float *> parameterMap {{"epsilon", &epsilon}};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const float *>(fc->fields[i].data);
        }
    }
    return new LayerNormPluginV4(name, epsilon);
}

IPluginV2 *LayerNormPluginV4Creator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new LayerNormPluginV4(name, serialData, serialLength);
}

void LayerNormPluginV4Creator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *LayerNormPluginV4Creator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *LayerNormPluginV4Creator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *LayerNormPluginV4Creator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION_V4;
}

const PluginFieldCollection *LayerNormPluginV4Creator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginV4Creator);

} // namespace nvinfer1
