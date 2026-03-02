/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
CuBLASGemmPlugin::CuBLASGemmPlugin(int const k, int const n, float const *w):
    mnK(k), mnN(n)
{
    WHERE_AM_I();
    mpCPUWeight.resize(k * n);
    std::copy(w, w + k * n, mpCPUWeight.begin());
}

CuBLASGemmPlugin::CuBLASGemmPlugin(CuBLASGemmPlugin const &p)
{
    WHERE_AM_I();
    mnK = p.mnK;
    mnN = p.mnN;
    mpCPUWeight.resize(mnK * mnN);
    std::copy(p.mpCPUWeight.begin(), p.mpCPUWeight.end(), mpCPUWeight.begin());
}

CuBLASGemmPlugin::~CuBLASGemmPlugin()
{
    WHERE_AM_I();
    if (mpGPUWeight)
    {
        cudaFree(mpGPUWeight);
    }
    if (mCuBLASHandle)
    {
        CHECK_CUBLAS(cublasDestroy(mCuBLASHandle));
    }
}

void CuBLASGemmPlugin::initializeContext()
{
    WHERE_AM_I();
    size_t nByte = sizeof(float) * mnK * mnN;
    cudaMalloc((void **)&mpGPUWeight, nByte);
    cudaMemcpy(mpGPUWeight, mpCPUWeight.data(), nByte, cudaMemcpyHostToDevice);
    CHECK_CUBLAS(cublasCreate(&mCuBLASHandle));
}

IPluginCapability *CuBLASGemmPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    WHERE_AM_I();
    switch (type)
    {
    case PluginCapabilityType::kBUILD:
        return static_cast<IPluginV3OneBuild *>(this);
    case PluginCapabilityType::kRUNTIME:
        return static_cast<IPluginV3OneRuntime *>(this);
    case PluginCapabilityType::kCORE:
        return static_cast<IPluginV3OneCore *>(this);
    }
    return nullptr;
}

CuBLASGemmPlugin *CuBLASGemmPlugin::clone() noexcept
{
    WHERE_AM_I();
    std::unique_ptr<CuBLASGemmPlugin> p {std::make_unique<CuBLASGemmPlugin>(*this)};
    return p.release();
}

char const *CuBLASGemmPlugin::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

char const *CuBLASGemmPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

char const *CuBLASGemmPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

int32_t CuBLASGemmPlugin::configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CuBLASGemmPlugin::getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t CuBLASGemmPlugin::getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs, int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    outputs[0].nbDims = inputs[0].nbDims;
    for (int i = 0; i < inputs[0].nbDims - 1; ++i)
    {
        outputs[0].d[i] = inputs[0].d[i];
    }
    outputs[0].d[outputs[0].nbDims - 1] = exprBuilder.constant(mnN);
    return 0;
}

bool CuBLASGemmPlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res {false};
    switch (pos)
    {
    case 0:
        res = inOut[0].desc.type == DataType::kFLOAT && inOut[0].desc.format == TensorFormat::kLINEAR;
        break;
    case 1:
        res = inOut[1].desc.type == inOut[0].desc.type && inOut[1].desc.format == inOut[0].desc.format;
        break;
    default: // should NOT be here!
        res = false;
    }
    PRINT_FORMAT_COMBINATION();
    return res;
}

int32_t CuBLASGemmPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

size_t CuBLASGemmPlugin::getWorkspaceSize(DynamicPluginTensorDesc const *inputs, int32_t nbInputs, DynamicPluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CuBLASGemmPlugin::getValidTactics(int32_t *tactics, int32_t nbTactics) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CuBLASGemmPlugin::getNbTactics() noexcept
{
    WHERE_AM_I();
    return 0;
}

char const *CuBLASGemmPlugin::getTimingCacheID() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t CuBLASGemmPlugin::getFormatCombinationLimit() noexcept
{
    WHERE_AM_I();
    return 1;
}

char const *CuBLASGemmPlugin::getMetadataString() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t CuBLASGemmPlugin::setTactic(int32_t tactic) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CuBLASGemmPlugin::onShapeChange(PluginTensorDesc const *in, int32_t nbInputs, PluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CuBLASGemmPlugin::enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int         nBatch = 1;
    float const alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        nBatch *= inputDesc[0].dims.d[i];
    }
    CHECK_CUBLAS(cublasSetStream(mCuBLASHandle, stream));
    CHECK_CUBLAS(cublasSgemm(mCuBLASHandle, CUBLAS_OP_N, CUBLAS_OP_N, mnN, nBatch, mnK, &alpha, mpGPUWeight, mnN, reinterpret_cast<float const *>(inputs[0]), mnK, &beta, reinterpret_cast<float *>(outputs[0]), mnN));
    return 0;
}

IPluginV3 *CuBLASGemmPlugin::attachToContext(IPluginResourceContext *context) noexcept
{
    WHERE_AM_I();

    CuBLASGemmPlugin *ret = this->clone();
    ret->initializeContext();
    return ret;
}

PluginFieldCollection const *CuBLASGemmPlugin::getFieldsToSerialize() noexcept
{
    WHERE_AM_I();
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(PluginField("K", &mnK, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("N", &mnN, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("Weight", mpCPUWeight.data(), PluginFieldType::kFLOAT32, mnK * mnN));
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields   = mDataToSerialize.data();
    return &mFCToSerialize;
}

CuBLASGemmPluginCreator::CuBLASGemmPluginCreator()
{
    WHERE_AM_I();
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("N", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("Weight", nullptr, PluginFieldType::kFLOAT32, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

char const *CuBLASGemmPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

char const *CuBLASGemmPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

PluginFieldCollection const *CuBLASGemmPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &mFC;
}

IPluginV3 *CuBLASGemmPluginCreator::createPlugin(char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept
{
    WHERE_AM_I();
    int          k, n;
    float const *w;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        std::string const field_name(fc->fields[i].name);
        if (field_name.compare("K") == 0)
        {
            k = *reinterpret_cast<int const *>(fc->fields[i].data);
        }
        else if (field_name.compare("N") == 0)
        {
            n = *reinterpret_cast<int const *>(fc->fields[i].data);
        }
        else if (field_name.compare("Weight") == 0)
        {
            w = reinterpret_cast<float const *>(fc->fields[i].data);
        }
    }
    return new CuBLASGemmPlugin(k, n, w);
}

char const *CuBLASGemmPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

REGISTER_TENSORRT_PLUGIN(CuBLASGemmPluginCreator);

} // namespace nvinfer1
