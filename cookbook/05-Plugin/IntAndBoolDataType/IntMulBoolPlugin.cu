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

#include "IntMulBoolPlugin.h"

// kernel for GPU
__global__ void kernel(const int *input0, const bool *input1, float *output, const int32_t nLength)
{
    for (int columnIndex = threadIdx.x; columnIndex < nLength; columnIndex += blockDim.x)
    {
        int32_t index = blockIdx.x * nLength + columnIndex;
        int     a     = input0[index];
        int     b     = input1[index];
        output[index] = float(a * int(b));
    }
}

namespace nvinfer1
{
// class IntMulBoolPlugin
IntMulBoolPlugin::IntMulBoolPlugin(const std::string &name):
    name_(name)
{
    WHERE_AM_I();
}

IntMulBoolPlugin::IntMulBoolPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
}

IntMulBoolPlugin::~IntMulBoolPlugin()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *IntMulBoolPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new IntMulBoolPlugin(name_);
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t IntMulBoolPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType IntMulBoolPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return DataType::kFLOAT;
}

DimsExprs IntMulBoolPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    return inputs[0];
}

bool IntMulBoolPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res;
    switch (pos)
    {
    case 0:
        res = inOut[0].type == DataType::kINT32 && inOut[0].format == TensorFormat::kLINEAR;
        break;
    case 1:
        res = inOut[1].type == DataType::kBOOL && inOut[0].format == TensorFormat::kLINEAR;
        break;
    case 2:
        res = inOut[pos].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
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

void IntMulBoolPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return;
}

size_t IntMulBoolPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t IntMulBoolPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int32_t nBatch  = inputDesc[0].dims.d[0];
    int32_t nLength = inputDesc[0].dims.d[1];

    kernel<<<nBatch, 256, 0, stream>>>(reinterpret_cast<const int *>(inputs[0]), reinterpret_cast<const bool *>(inputs[1]), reinterpret_cast<float *>(outputs[0]), nLength);

    return 0;
}

void IntMulBoolPlugin::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t IntMulBoolPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void IntMulBoolPlugin::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t IntMulBoolPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return 0;
}

void IntMulBoolPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    return;
}

void IntMulBoolPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *IntMulBoolPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *IntMulBoolPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *IntMulBoolPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

void IntMulBoolPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void IntMulBoolPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class IntMulBoolPluginCreator
PluginFieldCollection    IntMulBoolPluginCreator::fc_ {};
std::vector<PluginField> IntMulBoolPluginCreator::attr_;

IntMulBoolPluginCreator::IntMulBoolPluginCreator()
{
    WHERE_AM_I();
    attr_.clear();
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

IntMulBoolPluginCreator::~IntMulBoolPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *IntMulBoolPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    IntMulBoolPlugin *pObj = new IntMulBoolPlugin(name);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2DynamicExt *IntMulBoolPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    IntMulBoolPlugin *pObj = new IntMulBoolPlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void IntMulBoolPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *IntMulBoolPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *IntMulBoolPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *IntMulBoolPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

const PluginFieldCollection *IntMulBoolPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(IntMulBoolPluginCreator);

} // namespace nvinfer1
