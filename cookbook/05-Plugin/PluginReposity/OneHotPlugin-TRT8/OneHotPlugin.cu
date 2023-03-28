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

#include "OneHotPlugin.h"

/*  // V1
template<typename T>
__global__ void OneHotPluginKernel(int *pInput, T *pOutput, int nEmbedding, int nElement, int nBlockLoop)
{
    const int dstIndex = threadIdx.x;

    for (int i = 0; i < nBlockLoop; ++i)
    {
        int srcIndex = blockIdx.x + blockDim.x * i;
        if (srcIndex >= nElement || dstIndex >= nEmbedding)
        {
            return;
        }
        int srcValue                        = pInput[srcIndex];
        T   dstValue                        = (dstIndex == srcValue) ? T(1.0f) : T(0.0f);
        pOutput[srcIndex * nEmbedding + dstIndex] = dstValue;
    }
}

template<typename T>
__global__ void OneHotPluginBigKernel(int *pInput, T *pOutput, int nEmbedding, int nElement, int nBlockLoop, int nThreadLoop)
{
    for (int i = 0; i < nBlockLoop; ++i)
    {
        int srcIndex = blockIdx.x + blockDim.x * i;
        if (srcIndex >= nElement)
        {
            return;
        }
        int srcValue = pInput[srcIndex];
        for (int j = 0; j < nThreadLoop; ++j)
        {
            int dstIndex = blockDim.x * j + threadIdx.x;
            if (dstIndex >= nEmbedding)
            {
                return;
            }
            
            T   dstValue = (dstIndex == srcValue) ? T(1.0f) : T(0.0f);

            pOutput[srcIndex * nEmbedding + dstIndex] = dstValue;
        }
    }
}
*/

template<typename T>
__global__ void OneHotPluginKernel(int *pInput, T *pOutput, int nEmbedding, int nElement, int nBlockLoop) // nThreadLoop == 1 的情况
{
    const int stridePerBlockLoop = blockDim.x * nEmbedding;
    int       srcIndex = blockIdx.x - blockDim.x, dstIndex = srcIndex * nEmbedding + threadIdx.x;

    for (int i = 0; i < nBlockLoop; ++i)
    {
        srcIndex += blockDim.x;
        dstIndex += stridePerBlockLoop;
        if (srcIndex >= nElement || threadIdx.x >= nEmbedding)
        {
            return;
        }
        int srcValue      = pInput[srcIndex];
        T   dstValue      = (threadIdx.x == srcValue) ? T(1.0f) : T(0.0f);
        pOutput[dstIndex] = dstValue;
    }
}

template<typename T>
__global__ void OneHotPluginBigKernel(int *pInput, T *pOutput, int nEmbedding, int nElement, int nBlockLoop, int nThreadLoop)
{
    const int stridePerBlockLoop = blockDim.x * nEmbedding;
    int       srcIndex = blockIdx.x - blockDim.x, dstIndex = srcIndex * nEmbedding;

    for (int i = 0; i < nBlockLoop; ++i)
    {
        srcIndex += blockDim.x;
        dstIndex += stridePerBlockLoop;
        if (srcIndex >= nElement)
        {
            return;
        }
        int srcValue = pInput[srcIndex];

        int dstIndexWithinLine = threadIdx.x - blockDim.x;
        for (int j = 0; j < nThreadLoop; ++j)
        {
            dstIndexWithinLine += blockDim.x;
            if (dstIndexWithinLine >= nEmbedding)
            {
                return;
            }
            T dstValue                             = (dstIndexWithinLine == srcValue) ? T(1.0f) : T(0.0f);
            pOutput[dstIndex + dstIndexWithinLine] = dstValue;
        }
    }
}

namespace nvinfer1
{
// class OneHotPlugin
OneHotPlugin::OneHotPlugin(const std::string &name, int nEmbedding):
    name_(name)
{
    WHERE_AM_I();
    m_.nEmbedding = nEmbedding;
    m_.nBlockLoop = 1; // each block work for one element

    m_.blockDim    = nEmbedding <= maxTPB ? ALIGN_TO(nEmbedding, 32) : maxTPB;
    m_.nThreadLoop = CEIL_DIVIDE(m_.nEmbedding, m_.blockDim);
}

OneHotPlugin::OneHotPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

OneHotPlugin::~OneHotPlugin()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *OneHotPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new OneHotPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t OneHotPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType OneHotPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return DataType::kFLOAT;
}

DimsExprs OneHotPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs ret {inputs[0]};
    ret.nbDims += 1;
    ret.d[ret.nbDims - 1] = exprBuilder.constant(m_.nEmbedding);
    return ret;
}

bool OneHotPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    switch (pos)
    {
    case 0:
        return inOut[0].type == DataType::kINT32 && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
        return (inOut[1].type == DataType::kFLOAT || inOut[1].type == DataType::kHALF) && inOut[1].format == TensorFormat::kLINEAR;
    default: // should NOT be here!
        return false;
    }
    return false;
}

void OneHotPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return;
}

size_t OneHotPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t OneHotPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nElement = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        nElement *= inputDesc[0].dims.d[i];
    }

    int gridDim = CEIL_DIVIDE(nElement, m_.nBlockLoop);

    if (outputDesc[0].type == DataType::kHALF)
    {
        if (m_.nEmbedding <= maxTPB)
        {
            (OneHotPluginKernel<half>)<<<gridDim, m_.blockDim, 0, stream>>>((int *)inputs[0], (half *)outputs[0], m_.nEmbedding, nElement, m_.nBlockLoop);
        }
        else
        {
            (OneHotPluginBigKernel<half>)<<<gridDim, m_.blockDim, 0, stream>>>((int *)inputs[0], (half *)outputs[0], m_.nEmbedding, nElement, m_.nBlockLoop, m_.nThreadLoop);
        }
    }
    else
    {
        if (m_.nEmbedding <= maxTPB)
        {
            (OneHotPluginKernel<float>)<<<gridDim, m_.blockDim, 0, stream>>>((int *)inputs[0], (float *)outputs[0], m_.nEmbedding, nElement, m_.nBlockLoop);
        }
        else
        {
            (OneHotPluginBigKernel<float>)<<<gridDim, m_.blockDim, 0, stream>>>((int *)inputs[0], (float *)outputs[0], m_.nEmbedding, nElement, m_.nBlockLoop, m_.nThreadLoop);
        }
    }
    return 0;
}

void OneHotPlugin::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t OneHotPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void OneHotPlugin::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t OneHotPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void OneHotPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void OneHotPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *OneHotPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *OneHotPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *OneHotPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

void OneHotPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void OneHotPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class OneHotPluginCreator
PluginFieldCollection    OneHotPluginCreator::fc_ {};
std::vector<PluginField> OneHotPluginCreator::attr_;

OneHotPluginCreator::OneHotPluginCreator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("nEmbedding", nullptr, PluginFieldType::kINT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

OneHotPluginCreator::~OneHotPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2 *OneHotPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    int                          nEmbedding = 1;
    std::map<std::string, int *> parameterMap {{"nEmbedding", &nEmbedding}};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
    }
    OneHotPlugin *pObj = new OneHotPlugin(name, nEmbedding);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2 *OneHotPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    OneHotPlugin *pObj = new OneHotPlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void OneHotPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *OneHotPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *OneHotPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *OneHotPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

const PluginFieldCollection *OneHotPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(OneHotPluginCreator);

} // namespace nvinfer1
