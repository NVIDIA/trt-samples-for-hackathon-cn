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

#include "AddSubMulPlugin.h"

// kernel for GPU
__global__ void kernel0(const float *input0, const half *input1, float *output, const int32_t nLengthB)
{
    const int32_t iA = blockIdx.y * gridDim.x + blockIdx.x;
    const float   a  = input0[iA];

    for (int columnIndex = threadIdx.x; columnIndex < nLengthB; columnIndex += blockDim.x)
    {
        int32_t iB    = blockIdx.y * nLengthB + columnIndex;
        float   b     = float(input1[iB]);
        int     index = iA * nLengthB + columnIndex;
        output[index] = a * b;
    }
}

__global__ void kernel1(const float *input0, const half *input1, float *output, const int32_t nLengthA, const int32_t nLengthB)
{
    const int32_t nLength = (nLengthA < nLengthB) ? nLengthA : nLengthB;

    for (int columnIndex = threadIdx.x; columnIndex < nLength; columnIndex += blockDim.x)
    {
        float a       = input0[blockIdx.x * nLengthA + columnIndex];
        float b       = input1[blockIdx.x * nLengthB + columnIndex];
        int   index   = blockIdx.x * nLength + columnIndex;
        output[index] = a + b;
    }
}

namespace nvinfer1
{
// class AddSubMulPlugin
AddSubMulPlugin::AddSubMulPlugin(const std::string &name):
    name_(name)
{
    WHERE_AM_I();
}

AddSubMulPlugin::AddSubMulPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

AddSubMulPlugin::~AddSubMulPlugin()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *AddSubMulPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new AddSubMulPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t AddSubMulPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 2; // we have 2 outputs, A*B and A+B
}

DataType AddSubMulPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

DimsExprs AddSubMulPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs res;
    if (outputIndex == 0)
    {
        res.nbDims = 3;
        res.d[0]   = inputs[0].d[0];
        res.d[1]   = inputs[0].d[1];
        res.d[2]   = inputs[1].d[1];
    }
    else if (outputIndex == 1)
    {
        res.nbDims = 3;
        res.d[0]   = inputs[0].d[0];
        res.d[1]   = exprBuilder.constant(1);
        res.d[2]   = exprBuilder.operation(DimensionOperation::kMIN, *inputs[0].d[1], *inputs[1].d[1]);
    }
    else
    {
        printf("[AddSubMulPlugin::getOutputDimensions] outputIndex == %d\n", outputIndex);
    }
    return res;
}

bool AddSubMulPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res;
    switch (pos)
    {
    case 0:
        res = inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
        break;
    case 1:
        res = inOut[1].type == DataType::kHALF && inOut[0].format == TensorFormat::kLINEAR;
        break;
    case 2:
    case 3:
        res = inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
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

void AddSubMulPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    m_.nMaxBatchSize = in[0].max.d[0];
    m_.nMaxLengthA   = in[0].max.d[1];
    m_.nMaxLengthB   = in[1].max.d[1];
    return;
}

size_t AddSubMulPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    //return sizeof(float) * m_.nMaxBatchSize * m_.nMaxLengthA * m_.nMaxLengthB; // prepare GPU memory as the maximum size possible
    return sizeof(float) * inputs[0].dims.d[0] * inputs[0].dims.d[1] * inputs[1].dims.d[1]; // prepare GPU memory as the runtime value
}

int32_t AddSubMulPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int32_t nBatch   = inputDesc[0].dims.d[0];
    int32_t nLengthA = inputDesc[0].dims.d[1];
    int32_t nLengthB = inputDesc[1].dims.d[1];

    kernel0<<<dim3(nLengthA, nBatch), 256, 0, stream>>>(reinterpret_cast<const float *>(inputs[0]), reinterpret_cast<const half *>(inputs[1]), reinterpret_cast<float *>(workspace), nLengthB); // put the result in workspace and copy it into output[0] later in this example

    cudaMemcpyAsync(reinterpret_cast<float *>(outputs[0]), workspace, sizeof(float) * nBatch * nLengthA * nLengthB, cudaMemcpyDeviceToDevice, stream);

    kernel1<<<nBatch, 256, 0, stream>>>(reinterpret_cast<const float *>(inputs[0]), reinterpret_cast<const half *>(inputs[1]), reinterpret_cast<float *>(outputs[1]), nLengthA, nLengthB);

    return 0;
}

void AddSubMulPlugin::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t AddSubMulPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void AddSubMulPlugin::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t AddSubMulPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void AddSubMulPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void AddSubMulPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *AddSubMulPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *AddSubMulPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *AddSubMulPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

void AddSubMulPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void AddSubMulPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class AddSubMulPluginCreator
PluginFieldCollection    AddSubMulPluginCreator::fc_ {};
std::vector<PluginField> AddSubMulPluginCreator::attr_;

AddSubMulPluginCreator::AddSubMulPluginCreator()
{
    WHERE_AM_I();
    attr_.clear();
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

AddSubMulPluginCreator::~AddSubMulPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *AddSubMulPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    AddSubMulPlugin *pObj = new AddSubMulPlugin(name);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2DynamicExt *AddSubMulPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    AddSubMulPlugin *pObj = new AddSubMulPlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void AddSubMulPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *AddSubMulPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *AddSubMulPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *AddSubMulPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

const PluginFieldCollection *AddSubMulPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(AddSubMulPluginCreator);

} // namespace nvinfer1
