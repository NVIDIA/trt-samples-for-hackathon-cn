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

#include "AddScalarPlugin.h"

// 用于计算的 kernel
template<typename T>
__global__ void addScalarKernel(const T *input, T *output, const T scalar, const int nElement)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nElement)
        return;

    T _1          = input[index];
    T _2          = _1 + scalar;
    output[index] = _2;
}

namespace nvinfer1
{
// class AddScalarPlugin
AddScalarPlugin::AddScalarPlugin(const std::string &name, float scalar):
    name_(name)
{
    WHERE_AM_I()
    m_.scalar = scalar;
}

AddScalarPlugin::AddScalarPlugin(const std::string &name, const void *buffer, size_t length)
{
    WHERE_AM_I()
    memcpy(&m_, buffer, sizeof(m_));
}

AddScalarPlugin::~AddScalarPlugin()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *AddScalarPlugin::clone() const noexcept
{
    WHERE_AM_I()
    auto p = new AddScalarPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t AddScalarPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I()
    return 1;
}

DataType AddScalarPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I()
    return inputTypes[0];
}

DimsExprs AddScalarPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I()
    return inputs[0];
}

bool AddScalarPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I()
    bool res;
    switch (pos)
    {
    case 0:
        res = (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF) && inOut[0].format == TensorFormat::kLINEAR;
        break;
    case 1:
        res = inOut[1].format == inOut[0].format && inOut[1].type == inOut[0].type;
        break;
    default: // should NOT be here!
        res = false;
    }

#ifdef DEBUG
    std::cout << "\tpos=" << pos << ",res=" << res << "->[";
    for (int i = 0; i < nbInputs + nbOutputs; ++i)
    {
        std::cout << getFormatString(inOut[i].format) << ",";
    }
    std::cout << "],[";
    for (int i = 0; i < nbInputs + nbOutputs; ++i)
    {
        std::cout << getDataTypeString(inOut[i].type) << ",";
    }
    std::cout << "]" << std::endl;
#endif
    return res;
}

void AddScalarPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
}

size_t AddScalarPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I()
    return 0;
}

int32_t AddScalarPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I()
    int nElement = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
    {
        nElement *= inputDesc[0].dims.d[i];
    }
    dim3 grid(CEIL_DIVIDE(nElement, 256), 1, 1), block(256, 1, 1);

    if (inputDesc[0].type == DataType::kFLOAT)
    {
#ifdef DEBUG
        printf("\tFP32 kernel!\n");
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(40)); // 迫使样例程序在 FP16 模式下使用 FP16 的 kernel，实际使用时不需要这个等待
        (addScalarKernel<float>)<<<grid, block, 0, stream>>>(reinterpret_cast<const float *>(inputs[0]), reinterpret_cast<float *>(outputs[0]), m_.scalar, nElement);
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {
#ifdef DEBUG
        printf("\tFP16 kernel!\n");
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); // 迫使样例程序在 FP16 模式下使用 FP16 的 kernel，实际使用时不需要这个等待
        (addScalarKernel<__half>)<<<grid, block, 0, stream>>>(reinterpret_cast<const __half *>(inputs[0]), reinterpret_cast<__half *>(outputs[0]), __half(m_.scalar), nElement);
    }
    else
    {
        printf("\tUnsupport datatype!\n");
    }
    return 0;
}

void AddScalarPlugin::destroy() noexcept
{
    WHERE_AM_I();
    //delete this;
}

int32_t AddScalarPlugin::initialize() noexcept
{
    WHERE_AM_I()
    return 0;
}

void AddScalarPlugin::terminate() noexcept
{
    WHERE_AM_I();
}

size_t AddScalarPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I()
    return sizeof(m_);
}

void AddScalarPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I()
    memcpy(buffer, &m_, sizeof(m_));
}

void AddScalarPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I()
    namespace_ = pluginNamespace;
}
const char *AddScalarPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I()
    return namespace_.c_str();
}

const char *AddScalarPlugin::getPluginType() const noexcept
{
    WHERE_AM_I()
    return PLUGIN_NAME;
}

const char *AddScalarPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I()
    return PLUGIN_VERSION;
}

void AddScalarPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I()
}

void AddScalarPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
}

// class AddScalarPluginCreator
PluginFieldCollection    AddScalarPluginCreator::fc_ {};
std::vector<PluginField> AddScalarPluginCreator::attr_;

AddScalarPluginCreator::AddScalarPluginCreator()
{
    WHERE_AM_I()
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

AddScalarPluginCreator::~AddScalarPluginCreator()
{
    WHERE_AM_I();
}

// 最重要的两个成员函数，分别用于“接受参数创建 Plugin” 和 “去序列化创建 Plugin”
IPluginV2 *AddScalarPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I()
    float                          scalar = 0;
    std::map<std::string, float *> parameterMap {{"scalar", &scalar}};

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const float *>(fc->fields[i].data);
        }
    }
    return new AddScalarPlugin(name, *parameterMap["scalar"]);
}

IPluginV2 *AddScalarPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I()
    return new AddScalarPlugin(name, serialData, serialLength);
}

void AddScalarPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I()
    namespace_ = pluginNamespace;
}

const char *AddScalarPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I()
    return namespace_.c_str();
}

const char *AddScalarPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I()
    return PLUGIN_NAME;
}
const char *AddScalarPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I()
    return PLUGIN_VERSION;
}

const PluginFieldCollection *AddScalarPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I()
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(AddScalarPluginCreator);

} // namespace nvinfer1
