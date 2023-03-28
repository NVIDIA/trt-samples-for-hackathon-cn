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

#include "CuBLASGemmPlugin.h"

namespace nvinfer1
{
// class CuBLASGemmPlugin
CuBLASGemmPlugin::CuBLASGemmPlugin(const std::string &name, Weights weight, int k, int n, bool needDeepCopy = false):
    name_(name), bOwnWeight_(needDeepCopy), nK_(k), nN_(n)
{
    WHERE_AM_I();
    assert(weight.type == DataType::kFLOAT);
    assert(weight.values != nullptr);
    assert(weight.count == k * n);

    weight_.type  = DataType::kFLOAT;
    weight_.count = weight.count;
    if (needDeepCopy)
    {
        size_t size    = sizeof(float) * weight.count;
        weight_.values = malloc(size);
        memcpy(reinterpret_cast<char *>(const_cast<void *>(weight_.values)), weight.values, size);
    }
    else
    {
        weight_.values = weight.values;
    }

    CHECK(cublasCreate(&handle_));
}

CuBLASGemmPlugin::CuBLASGemmPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name), bOwnWeight_(true)
{
    WHERE_AM_I();
    const char *data   = reinterpret_cast<const char *>(buffer);
    size_t      offset = 0;
    memcpy(&nK_, data + offset, sizeof(nK_));
    offset += sizeof(nK_);
    memcpy(&nN_, data + offset, sizeof(nN_));
    offset += sizeof(nN_);

    weight_.type   = DataType::kFLOAT;
    weight_.count  = nK_ * nN_;
    size_t size    = sizeof(float) * nK_ * nN_;
    weight_.values = malloc(size);
    memcpy(reinterpret_cast<char *>(const_cast<void *>(weight_.values)), data + offset, size);

    CHECK(cublasCreate(&handle_));
}

CuBLASGemmPlugin::~CuBLASGemmPlugin()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *CuBLASGemmPlugin::clone() const noexcept
{
    WHERE_AM_I();
    CuBLASGemmPlugin *p = new CuBLASGemmPlugin(name_, weight_, nK_, nN_, false);
    p->setPluginNamespace(namespace_.c_str());
    p->pGPUWeight_ = this->pGPUWeight_;
    return p;
}

int32_t CuBLASGemmPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType CuBLASGemmPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

DimsExprs CuBLASGemmPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs ret {inputs[0]};
    ret.d[inputs[0].nbDims - 1] = exprBuilder.constant(nN_);
    return ret;
}

bool CuBLASGemmPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
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

void CuBLASGemmPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return;
}

size_t
CuBLASGemmPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CuBLASGemmPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int         nBatch = 1;
    const float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        nBatch *= inputDesc[0].dims.d[i];
    }

    CHECK(cublasSetStream(handle_, stream));
    CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, nN_, nBatch, nK_, &alpha, pGPUWeight_, nN_, (const float *)inputs[0], nK_, &beta, (float *)outputs[0], nN_));
    return 0;
}

int32_t CuBLASGemmPlugin::initialize() noexcept
{
    WHERE_AM_I();
    size_t size = sizeof(float) * weight_.count;
    CHECK(cudaMalloc((void **)&pGPUWeight_, size));
    CHECK(cudaMemcpy(pGPUWeight_, weight_.values, size, cudaMemcpyHostToDevice));
    return 0;
}

void CuBLASGemmPlugin::terminate() noexcept
{
    CHECK(cudaFree(pGPUWeight_));
    WHERE_AM_I();
    return;
}

void CuBLASGemmPlugin::destroy() noexcept
{
    WHERE_AM_I();
    if (bOwnWeight_)
    {
        free(const_cast<void *>(weight_.values));
    }
    CHECK(cublasDestroy(handle_));
    return;
}

size_t CuBLASGemmPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(nK_) + sizeof(nN_) + sizeof(float) * weight_.count;
}

void CuBLASGemmPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    char * data   = reinterpret_cast<char *>(buffer);
    size_t offset = 0;
    memcpy(data + offset, &nK_, sizeof(nK_));
    offset += sizeof(nK_);
    memcpy(data + offset, &nN_, sizeof(nN_));
    offset += sizeof(nN_);
    size_t size = sizeof(float) * nK_ * nN_;
    memcpy(data + offset, weight_.values, size);
    return;
}

void CuBLASGemmPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *CuBLASGemmPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *CuBLASGemmPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *CuBLASGemmPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

void CuBLASGemmPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    //handle_ = contextCublas;
    return;
}

void CuBLASGemmPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class CuBLASGemmPluginCreator
PluginFieldCollection    CuBLASGemmPluginCreator::fc_ {};
std::vector<PluginField> CuBLASGemmPluginCreator::attr_;

CuBLASGemmPluginCreator::CuBLASGemmPluginCreator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("scalar", nullptr, PluginFieldType::kFLOAT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

CuBLASGemmPluginCreator::~CuBLASGemmPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *CuBLASGemmPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    int     k, n;
    Weights w;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        PluginField field = fc->fields[i];
        std::string field_name(field.name);

        if (field_name.compare("weight") == 0)
        {
            w.values = field.data;
            w.count  = field.length;
            w.type   = DataType::kFLOAT;
            continue;
        }
        if (field_name.compare("k") == 0)
        {
            k = *reinterpret_cast<const int *>(field.data);
        }
        if (field_name.compare("n") == 0)
        {
            n = *reinterpret_cast<const int *>(field.data);
        }
    }
    return new CuBLASGemmPlugin(name, w, k, n, true);
}

IPluginV2DynamicExt *CuBLASGemmPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new CuBLASGemmPlugin(name, serialData, serialLength);
}

void CuBLASGemmPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *CuBLASGemmPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *CuBLASGemmPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *CuBLASGemmPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

const PluginFieldCollection *CuBLASGemmPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(CuBLASGemmPluginCreator);

} // namespace nvinfer1
