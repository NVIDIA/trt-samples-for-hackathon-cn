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

#include "LoadNpzPlugin.h"

namespace nvinfer1
{
// class LoadNpzPlugin
LoadNpzPlugin::LoadNpzPlugin(const std::string &name, bool bOwnWeight, float *pCPU, float *pGPU):
    name_(name), bOwnWeight_(bOwnWeight)
{
    WHERE_AM_I_LOADNPZ();
    if (bOwnWeight)
    {
        cnpy::npz_t    npzFile = cnpy::npz_load(dataFile);
        cnpy::NpyArray array   = npzFile[dataName];

        pCPU_ = (float *)malloc(sizeof(float) * nDataElement);
        memcpy(pCPU_, array.data<float>(), sizeof(float) * nDataElement);
    }
    else
    {
        this->pCPU_ = pCPU;
        this->pGPU_ = pGPU;
    }
}

LoadNpzPlugin::LoadNpzPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name), bOwnWeight_(true)
{
    WHERE_AM_I_LOADNPZ();
    cnpy::npz_t    npzFile = cnpy::npz_load(dataFile);
    cnpy::NpyArray array   = npzFile[dataName];

    pCPU_ = (float *)malloc(sizeof(float) * nDataElement);
    memcpy(pCPU_, array.data<float>(), sizeof(float) * nDataElement);
}

LoadNpzPlugin::~LoadNpzPlugin()
{
    WHERE_AM_I_LOADNPZ();
    if (bOwnWeight_)
    {
        free(pCPU_);
    }
}

IPluginV2DynamicExt *LoadNpzPlugin::clone() const noexcept
{
    WHERE_AM_I_LOADNPZ();
    auto p = new LoadNpzPlugin(name_, false, this->pCPU_, this->pGPU_);
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t LoadNpzPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I_LOADNPZ();
    return 1;
}

DataType LoadNpzPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I_LOADNPZ();
    return DataType::kFLOAT;
}

DimsExprs LoadNpzPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I_LOADNPZ();
    DimsExprs ret;
    ret.nbDims = 4;
    for (int i = 0; i < ret.nbDims; ++i)
    {
        ret.d[i] = exprBuilder.constant(4);
    }
    return ret;
}

bool LoadNpzPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I_LOADNPZ();
    bool res;
    switch (pos)
    {
    case 0:
        res = inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
        break;
    case 1:
        res = inOut[1].type == DataType::kFLOAT && inOut[1].format == TensorFormat::kLINEAR;
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

void LoadNpzPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I_LOADNPZ();
    return;
}

size_t LoadNpzPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I_LOADNPZ();
    return 0;
}

int32_t LoadNpzPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I_LOADNPZ();
    cudaMemcpyAsync(outputs[0], pGPU_, sizeof(float) * nDataElement, cudaMemcpyDeviceToDevice, stream);
    return 0;
}

int32_t LoadNpzPlugin::initialize() noexcept
{
    WHERE_AM_I_LOADNPZ();
    cudaMalloc(&pGPU_, sizeof(float) * nDataElement);
    cudaMemcpy(pGPU_, pCPU_, sizeof(float) * nDataElement, cudaMemcpyHostToDevice);
    return 0;
}

void LoadNpzPlugin::terminate() noexcept
{
    WHERE_AM_I_LOADNPZ();
    if (bOwnWeight_)
    {
        cudaFree(pGPU_);
    }
    return;
}

void LoadNpzPlugin::destroy() noexcept
{
    WHERE_AM_I_LOADNPZ();
    return;
}

size_t LoadNpzPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I_LOADNPZ();
    return 0;
}

void LoadNpzPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I_LOADNPZ();
    return;
}

void LoadNpzPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I_LOADNPZ();
    namespace_ = pluginNamespace;
    return;
}

const char *LoadNpzPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I_LOADNPZ();
    return namespace_.c_str();
}

const char *LoadNpzPlugin::getPluginType() const noexcept
{
    WHERE_AM_I_LOADNPZ();
    return PLUGIN_NAME;
}

const char *LoadNpzPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I_LOADNPZ();
    return PLUGIN_VERSION;
}

void LoadNpzPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I_LOADNPZ();
    return;
}

void LoadNpzPlugin::detachFromContext() noexcept
{
    WHERE_AM_I_LOADNPZ();
    return;
}

// class LoadNpzPluginCreator
PluginFieldCollection    LoadNpzPluginCreator::fc_ {};
std::vector<PluginField> LoadNpzPluginCreator::attr_;

LoadNpzPluginCreator::LoadNpzPluginCreator()
{
    WHERE_AM_I();
    attr_.clear();
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

LoadNpzPluginCreator::~LoadNpzPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *LoadNpzPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    LoadNpzPlugin *pObj = new LoadNpzPlugin(name, true, nullptr, nullptr);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2DynamicExt *LoadNpzPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    LoadNpzPlugin *pObj = new LoadNpzPlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void LoadNpzPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *LoadNpzPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *LoadNpzPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *LoadNpzPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

const PluginFieldCollection *LoadNpzPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(LoadNpzPluginCreator);

} // namespace nvinfer1
