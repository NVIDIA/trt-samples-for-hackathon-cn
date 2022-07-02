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

struct mySum
{
    __host__ __device__ __forceinline__ float2 operator()(const float2 &a, const float2 &b) const
    {
        return make_float2(a.x + b.x, a.y + b.y);
    }
};

template<typename T, int TPB, int VPT>
__global__ void layerNormKernel(const T *input, const T *gamma, const T *beta, T *output)
{
    const int   idx = blockIdx.x * 256 + VPT * threadIdx.x;
    T           localX[VPT], localGamma[VPT], localBeta[VPT];
    float2      localFloat2 = {0.f, 0.f};
    const float denominator = float(1) / float(256);

    copy<sizeof(T) * VPT>(&input[idx], localX);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const float tmp = denominator * (float)localX[it];
        localFloat2.x += tmp;
        localFloat2.y += tmp * (float)localX[it];
    }

    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], localGamma);
    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], localBeta);

    using BlockReduce = cub::BlockReduce<float2, TPB>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float                             mu;     // mean
    __shared__ float                             rsigma; // 1 / std.dev.

    const float2 sumKV = BlockReduce(temp).Reduce(localFloat2, mySum());

    if (threadIdx.x == 0)
    {
        mu     = sumKV.x;
        rsigma = rsqrt(sumKV.y - mu * mu + 1e-6);
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        localX[it] = ((float)localX[it] - mu) * rsigma * (float)localGamma[it] + (float)localBeta[it];
    }

    copy<sizeof(T) * VPT>(localX, &output[idx]);
}

template __global__ void layerNormKernel<float, 64, 4>(const float *, const float *, const float *, float *);
template __global__ void layerNormKernel<half, 32, 8>(const half *, const half *, const half *, half *);

namespace nvinfer1
{
// class LayerNormPlugin
LayerNormPlugin::LayerNormPlugin(const std::string &name, float epsilon):
    name_(name)
{
    WHERE_AM_I();
    m_.epsilon = epsilon;
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

    if (inputDesc[0].type == DataType::kFLOAT)
    {
        constexpr int VPT = 16 / sizeof(float);
        constexpr int TPB = 256 / VPT;
        (layerNormKernel<float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>((const float *)inputs[0], (const float *)inputs[1], (const float *)inputs[2], (float *)outputs[0]);
    }
    else
    {
        constexpr int VPT = 16 / sizeof(half);
        constexpr int TPB = 256 / VPT;
        (layerNormKernel<half, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>((const half *)inputs[0], (const half *)inputs[1], (const half *)inputs[2], (half *)outputs[0]);
    }
    return 0;
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
    attr_.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

LayerNormPluginCreator::~LayerNormPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2 *LayerNormPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
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
    return new LayerNormPlugin(name, epsilon);
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

const PluginFieldCollection *LayerNormPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

} // namespace nvinfer1
