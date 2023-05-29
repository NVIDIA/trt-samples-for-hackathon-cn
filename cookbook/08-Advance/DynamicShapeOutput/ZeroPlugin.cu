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

#include "ZeroPlugin.h"

// kernel for GPU
__global__ void ZeroKernel(const float *input, float *output, const float scalar, const int nElement)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nElement)
        return;

    float _1      = input[index];
    float _2      = _1 + scalar;
    output[index] = _2;
}

namespace nvinfer1
{
// class ZeroPlugin
ZeroPlugin::ZeroPlugin(const std::string &name):
    name_(name)
{
    WHERE_AM_I();
}

ZeroPlugin::ZeroPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
}

ZeroPlugin::~ZeroPlugin()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *ZeroPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new ZeroPlugin(name_, nullptr, 0);
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t ZeroPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType ZeroPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

DimsExprs ZeroPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs res = inputs[0];
    res.nbDims    = 2;
    return res;
}

bool ZeroPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res;
    switch (pos)
    {
    case 0:
        res = inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
        break;
    case 1:
        res = inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
        break;
    default: // should NOT be here!
        res = false;
    }
#ifdef DEBUG
    std::cout << "\tpos=" << pos << ",res=" << res << "->[";
    for (int i = 0; i < nbInputs + nbOutputs; ++i)
    {
        std::cout << formatToString(inOut[i].format) << ",";
    }
    std::cout << "],[";
    for (int i = 0; i < nbInputs + nbOutputs; ++i)
    {
        std::cout << dataTypeToString(inOut[i].type) << ",";
    }
    std::cout << "]" << std::endl;
#endif
    return res;
}

void ZeroPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return;
}

size_t ZeroPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t ZeroPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nElement = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        nElement *= inputDesc[0].dims.d[i];
    }
    cudaMemsetAsync((float *)outputs[0], 0, sizeof(float) * nElement, stream);
    return 0;
}

void ZeroPlugin::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t ZeroPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void ZeroPlugin::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t ZeroPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return 0;
}

void ZeroPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    return;
}

void ZeroPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *ZeroPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *ZeroPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *ZeroPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

void ZeroPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void ZeroPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class ZeroPluginCreator
PluginFieldCollection    ZeroPluginCreator::fc_ {};
std::vector<PluginField> ZeroPluginCreator::attr_;

ZeroPluginCreator::ZeroPluginCreator()
{
    WHERE_AM_I();
    attr_.clear();
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

ZeroPluginCreator::~ZeroPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *ZeroPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    ZeroPlugin *pObj = new ZeroPlugin(name);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2DynamicExt *ZeroPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    ZeroPlugin *pObj = new ZeroPlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void ZeroPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *ZeroPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *ZeroPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *ZeroPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

const PluginFieldCollection *ZeroPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(ZeroPluginCreator);

} // namespace nvinfer1