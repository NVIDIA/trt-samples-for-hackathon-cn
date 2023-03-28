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

#include "LayerNormPluginOneFlow.h"

// ALIGNPTR
int8_t *alignPtr(int8_t *ptr, uintptr_t to)
{
    uintptr_t addr = (uintptr_t)ptr;
    if (addr % to)
    {
        addr += to - addr % to;
    }
    return (int8_t *)addr;
}

// NEXTWORKSPACEPTR
int8_t *nextWorkspacePtr(int8_t *ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t)ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t *)addr, CUDA_MEM_ALIGN);
}

namespace nvinfer1
{
// class LayerNormPluginV5
LayerNormPluginV5::LayerNormPluginV5(const std::string &name, float epsilon):
    name_(name)
{
    WHERE_AM_I();
    m_.epsilon = epsilon;
}

LayerNormPluginV5::LayerNormPluginV5(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

LayerNormPluginV5::~LayerNormPluginV5()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *LayerNormPluginV5::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new LayerNormPluginV5(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t LayerNormPluginV5::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType LayerNormPluginV5::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return DataType::kFLOAT;
}

DimsExprs LayerNormPluginV5::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    return inputs[0];
}

bool LayerNormPluginV5::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    switch (pos)
    {
    case 0:
    case 1:
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) && inOut[pos].format == TensorFormat::kLINEAR;
    default: // should NOT be here!
        return false;
    }
    return false;
}

void LayerNormPluginV5::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return;
}

size_t LayerNormPluginV5::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    int          nBlock        = inputs[0].dims.d[0] * inputs[0].dims.d[1];
    const size_t element_size  = (inputs[0].type == DataType::kFLOAT) ? sizeof(float) : sizeof(__half);
    size_t       workspaceSize = ALIGN_TO(nBlock * element_size, CUDA_MEM_ALIGN) * 2;
    //realSize = nBlock * element_size;
    //workspaceSize += ALIGN_TO(realSize, CUDA_MEM_ALIGN);
    return workspaceSize;
}

int32_t LayerNormPluginV5::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    // #rows
    int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    // #cols
    int nValuePerBlock = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

    auto *    mean              = reinterpret_cast<float *>(workspace);
    uintptr_t mean_size         = ALIGN_TO(nBlock * sizeof(float), CUDA_MEM_ALIGN);
    auto *    inv_variance      = reinterpret_cast<float *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(mean), mean_size));
    uintptr_t inv_variance_size = mean_size;
    if (inputDesc[0].type == DataType::kFLOAT && outputDesc[0].type == DataType::kFLOAT)
    {
        oneflow::cuda::layer_norm::DirectLoad<float, float>  load((float *)inputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DirectStore<float, float> store((float *)outputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), float>(stream, load, store, nBlock, nValuePerBlock, m_.epsilon, mean, inv_variance);
    }
    else if (inputDesc[0].type == DataType::kFLOAT && outputDesc[0].type == DataType::kHALF)
    {
        oneflow::cuda::layer_norm::DirectLoad<float, float> load((float *)inputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DirectStore<float, half> store((half *)outputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), float>(stream, load, store, nBlock, nValuePerBlock, m_.epsilon, mean, inv_variance);
    }
    else if (inputDesc[0].type == DataType::kHALF && outputDesc[0].type == DataType::kFLOAT)
    {
        oneflow::cuda::layer_norm::DirectLoad<half, float>   load((half *)inputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DirectStore<float, float> store((float *)outputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), float>(stream, load, store, nBlock, nValuePerBlock, m_.epsilon, mean, inv_variance);
    }
    else if (inputDesc[0].type == DataType::kHALF && outputDesc[0].type == DataType::kHALF)
    {
        oneflow::cuda::layer_norm::DirectLoad<half, float>  load((half *)inputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DirectStore<float, half> store((half *)outputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), float>(stream, load, store, nBlock, nValuePerBlock, m_.epsilon, mean, inv_variance);
    }
    else
    {
        printf("[LayerNormPluginV5 ERROR] Should never reach here\n");
    }
    return 0;
}

void LayerNormPluginV5::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t LayerNormPluginV5::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void LayerNormPluginV5::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t LayerNormPluginV5::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void LayerNormPluginV5::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void LayerNormPluginV5::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *LayerNormPluginV5::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *LayerNormPluginV5::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *LayerNormPluginV5::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

void LayerNormPluginV5::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void LayerNormPluginV5::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class LayerNormPluginV5Creator
PluginFieldCollection    LayerNormPluginV5Creator::fc_ {};
std::vector<PluginField> LayerNormPluginV5Creator::attr_;

LayerNormPluginV5Creator::LayerNormPluginV5Creator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

LayerNormPluginV5Creator::~LayerNormPluginV5Creator()
{
    WHERE_AM_I();
}

IPluginV2 *LayerNormPluginV5Creator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
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
    return new LayerNormPluginV5(name, epsilon);
}

IPluginV2 *LayerNormPluginV5Creator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new LayerNormPluginV5(name, serialData, serialLength);
}

void LayerNormPluginV5Creator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *LayerNormPluginV5Creator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *LayerNormPluginV5Creator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *LayerNormPluginV5Creator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

const PluginFieldCollection *LayerNormPluginV5Creator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginV5Creator);

} // namespace nvinfer1
