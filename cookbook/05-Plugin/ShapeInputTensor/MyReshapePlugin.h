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
static char const *PLUGIN_NAME {"MyReshape"};
static char const *PLUGIN_NAMESPACE {""};
static char const *PLUGIN_VERSION {"1"};
} // namespace

namespace nvinfer1
{
class MyReshapePlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{
private:
    std::vector<PluginField> mDataToSerialize;
    PluginFieldCollection    mFCToSerialize;

public:
    MyReshapePlugin();
    MyReshapePlugin(MyReshapePlugin const &p) = default;
    void initFieldsToSerialize();

    // IPluginV3 methods
    IPluginCapability *getCapabilityInterface(PluginCapabilityType type) noexcept override;
    IPluginV3         *clone() noexcept override;

    // IPluginV3OneCore methods
    char const *getPluginName() const noexcept override;
    char const *getPluginVersion() const noexcept override;
    char const *getPluginNamespace() const noexcept override;

    // IPluginV3OneBuild methods
    int32_t     configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept override;
    int32_t     getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs, DataType const *inputTypes, int32_t nbInputs) const noexcept override;
    int32_t     getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs, int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept override;
    bool        supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t     getNbOutputs() const noexcept override;
    size_t      getWorkspaceSize(DynamicPluginTensorDesc const *inputs, int32_t nbInputs, DynamicPluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept override;
    int32_t     getValidTactics(int32_t *tactics, int32_t nbTactics) noexcept override;
    int32_t     getNbTactics() noexcept override;
    char const *getTimingCacheID() noexcept override;
    int32_t     getFormatCombinationLimit() noexcept override;
    char const *getMetadataString() noexcept override;

    // IPluginV3OneRuntime methods
    int32_t                      setTactic(int32_t tactic) noexcept override;
    int32_t                      onShapeChange(PluginTensorDesc const *in, int32_t nbInputs, PluginTensorDesc const *out, int32_t nbOutputs) noexcept override;
    int32_t                      enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
    IPluginV3                   *attachToContext(IPluginResourceContext *context) noexcept override;
    PluginFieldCollection const *getFieldsToSerialize() noexcept override;
};

class MyReshapePluginCreator : public IPluginCreatorV3One
{
private:
    nvinfer1::PluginFieldCollection    mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;

public:
    MyReshapePluginCreator();
    char const                  *getPluginName() const noexcept override;
    char const                  *getPluginVersion() const noexcept override;
    PluginFieldCollection const *getFieldNames() noexcept override;
    IPluginV3                   *createPlugin(char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept override;
    char const                  *getPluginNamespace() const noexcept override;
};

} // namespace nvinfer1
