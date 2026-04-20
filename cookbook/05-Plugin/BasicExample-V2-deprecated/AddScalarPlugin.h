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

#include "cookbookHelper.cuh"

namespace
{
static char const *PLUGIN_NAME {"AddScalar"};
static char const *PLUGIN_NAMESPACE {""};
static char const *PLUGIN_VERSION {"1"};
} // namespace

namespace nvinfer1
{

class AddScalarPlugin : public IPluginV2DynamicExt
{
private:
    std::string mNamespace {PLUGIN_NAMESPACE};
    struct
    {
        float scalar;
    } m;

public:
    AddScalarPlugin() = delete;
    AddScalarPlugin(float const scalar);
    AddScalarPlugin(void const *buffer, size_t const length);
    ~AddScalarPlugin();

    // IPluginV2 methods
    char const *getPluginType() const noexcept override;
    char const *getPluginVersion() const noexcept override;
    int32_t     getNbOutputs() const noexcept override;
    int32_t     initialize() noexcept override;
    void        terminate() noexcept override;
    size_t      getSerializationSize() const noexcept override;
    void        serialize(void *buffer) const noexcept override;
    void        destroy() noexcept override;
    void        setPluginNamespace(char const *pluginNamespace) noexcept override;
    char const *getPluginNamespace() const noexcept override;

    // IPluginV2Ext methods
    DataType getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept override;
    void     attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept override;
    void     detachFromContext() noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt *clone() const noexcept override;
    DimsExprs            getOutputDimensions(int32_t outputIndex, DimsExprs const *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;
    bool                 supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void                 configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept override;
    size_t               getWorkspaceSize(PluginTensorDesc const *inputs, int32_t nbInputs, PluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept override;
    int32_t              enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

protected:
    // To prevent compiler warnings
    using nvinfer1::IPluginV2::enqueue;
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
    using nvinfer1::IPluginV2Ext::configurePlugin;
};

class AddScalarPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection    mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string                     mNamespace {PLUGIN_NAMESPACE};

public:
    AddScalarPluginCreator();
    ~AddScalarPluginCreator();
    char const                  *getPluginName() const noexcept override;
    char const                  *getPluginVersion() const noexcept override;
    PluginFieldCollection const *getFieldNames() noexcept override;
    IPluginV2DynamicExt         *createPlugin(char const *name, PluginFieldCollection const *fc) noexcept override;
    IPluginV2DynamicExt         *deserializePlugin(char const *name, void const *serialData, size_t serialLength) noexcept override;
    void                         setPluginNamespace(char const *pluginNamespace) noexcept override;
    char const                  *getPluginNamespace() const noexcept override;
};

} // namespace nvinfer1
