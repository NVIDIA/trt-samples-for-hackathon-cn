/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_GROUPNORM_PLUGIN_H
#define TRT_GROUPNORM_PLUGIN_H

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "common/plugin.h"

#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>
#include <cassert>
#include <cuda.h>
#include <cuda_fp16.h>


namespace nvinfer1
{
namespace plugin
{

static std::string const kGATHER_V2_PLUGIN_NAME{"GatherV2"};
static std::string const kGATHER_V2_PLUGIN_VERSION{"1"};

class GatherV2Plugin : public IPluginV2DynamicExt
{
public:
    GatherV2Plugin() = default;
    GatherV2Plugin(std::string name, void const* buffer, size_t length)
            : mName(std::move(name)) {
        assert(buffer != nullptr);
        assert(length == 0);

        auto const* d = static_cast<char const*>(buffer);
        auto const* a = d;

        assert(d == a + length);
    }
    ~GatherV2Plugin() override = default;

    // Methods inherited from IPluginV2
    char const* getPluginType() const noexcept override { return kGATHER_V2_PLUGIN_NAME.c_str(); }
    char const* getPluginVersion() const noexcept override { return kGATHER_V2_PLUGIN_VERSION.c_str(); }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void* buffer) const noexcept override {
        assert(buffer != nullptr);
        auto* d = static_cast<char*>(buffer);
        auto* a = d;
        assert(d == a + getSerializationSize());
    }
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(char const* pluginNamespace) noexcept override { mNameSpace = pluginNamespace; }
    char const* getPluginNamespace() const noexcept override { return mNameSpace.c_str(); }

    // Method inherited from IPluginV2Ext
    DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept override { return DataType::kFLOAT; } //inputTypes[0]

    // Methods inherited from IPluginV2DynamicExt
    IPluginV2DynamicExt* clone() const noexcept override {
        auto p = new GatherV2Plugin(*this);
        p->setPluginNamespace(mNameSpace.c_str());
        return p;
    }
    DimsExprs getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override {
        auto inputDims = inputs[0];
        auto indicesDims = inputs[1];
        indicesDims.nbDims = 3;
        indicesDims.d[2] = inputDims.d[1];
        return indicesDims;
    }
    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override {
        PLUGIN_ASSERT(pos >= 0 && pos <= 2);
        if (pos == 1)
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kINT32;
        return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
    }
    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override {}
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override { return 0; }
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    std::string mName;
    std::string mNameSpace;
};

class GatherV2PluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    GatherV2PluginCreator() {
        mPluginAttributes.clear();
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }
    ~GatherV2PluginCreator() override = default;

    char const* getPluginName() const noexcept override { return kGATHER_V2_PLUGIN_NAME.c_str(); }
    char const* getPluginVersion() const noexcept override { return kGATHER_V2_PLUGIN_VERSION.c_str(); }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }
    IPluginV2* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override {
        return new GatherV2Plugin();
    }
    IPluginV2* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override {
        return new GatherV2Plugin(name, serialData, serialLength);
    }

private:
    PluginFieldCollection mFC{};
    std::vector<PluginField> mPluginAttributes;
    std::string mNameSpace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_GROUPNORM_PLUGIN_H
