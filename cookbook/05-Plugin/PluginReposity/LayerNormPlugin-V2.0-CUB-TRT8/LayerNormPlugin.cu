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

template<typename T, int n>
__global__ void layerNormKernel(T *pInput, T *pOutput, float epsilon)
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
        return inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
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
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1], nValuePerBlock = inputDesc[0].dims.d[2];

    if (inputDesc[0].type == DataType::kFLOAT)
    {
        switch (nValuePerBlock)
        {
        case 256: // 仅用于处理 nEmbedding 为 256 的情况
            (layerNormKernel<float, 256>)<<<nBlock, nValuePerBlock, 0, stream>>>((float *)inputs[0], (float *)outputs[0], m_.epsilon);
            break;
        default: // should NOT be here!
            printf("[LayerNormPlugin::enqueue] nValuePerBlock = %d is not supported\n", nValuePerBlock);
            break;
        }
    }
    else
    {
        switch (nValuePerBlock)
        {
        case 256: // 仅用于处理 nEmbedding 为 256 的情况
            (layerNormKernel<half, 256>)<<<nBlock, nValuePerBlock, 0, stream>>>((half *)inputs[0], (half *)outputs[0], m_.epsilon);
            break;
        default: // should NOT be here!
            printf("[LayerNormPlugin::enqueue] nValuePerBlock = %d is not supported\n", nValuePerBlock);
            break;
        }
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
