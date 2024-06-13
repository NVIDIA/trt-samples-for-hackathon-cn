/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

// kernel for GPU
__global__ void addScalarKernel(const float *input, float *output, float const scalar, int const nElement)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nElement)
        return;

    float _1      = input[index];
    float _2      = _1 + scalar;
    output[index] = _2;
}

namespace nvinfer1
{
AddScalarPlugin::AddScalarPlugin(float const scalar):
    mScalar(scalar)
{
    WHERE_AM_I();
    initFieldsToSerialize();
}

void AddScalarPlugin::initFieldsToSerialize()
{
    WHERE_AM_I();
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(PluginField("scalar", &mScalar, PluginFieldType::kFLOAT32, 1));
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields   = mDataToSerialize.data();
}

IPluginCapability *AddScalarPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    WHERE_AM_I();
    mPluginCapabilityType = type;
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

IPluginV3 *AddScalarPlugin::clone() noexcept
{
    WHERE_AM_I();
    std::unique_ptr<AddScalarPlugin> p {std::make_unique<AddScalarPlugin>(*this)};
    p->initFieldsToSerialize();
    return p.release();
}

char const *AddScalarPlugin::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

char const *AddScalarPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

char const *AddScalarPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

int32_t AddScalarPlugin::configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t AddScalarPlugin::getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t AddScalarPlugin::getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs, int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    outputs[0].nbDims = inputs[0].nbDims;
    for (int i = 0; i < outputs[0].nbDims; ++i)
    {
        outputs[0].d[i] = inputs[0].d[i];
    }
    return 0;
}

bool AddScalarPlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
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

int32_t AddScalarPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

size_t AddScalarPlugin::getWorkspaceSize(DynamicPluginTensorDesc const *inputs, int32_t nbInputs, DynamicPluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t AddScalarPlugin::getValidTactics(int32_t *tactics, int32_t nbTactics) noexcept
{
    WHERE_AM_I();
    tactics[0] = int32_t(Tactic::kNaive);
    tactics[1] = int32_t(Tactic::kSmart);
    return 0;
}

int32_t AddScalarPlugin::getNbTactics() noexcept
{
    WHERE_AM_I();
    return N_TACTIC;
}

char const *AddScalarPlugin::getTimingCacheID() noexcept
{
    WHERE_AM_I();
    return "_AddScalarPlugin";
}

int32_t AddScalarPlugin::getFormatCombinationLimit() noexcept
{
    WHERE_AM_I();
    return 1;
}

char const *AddScalarPlugin::getMetadataString() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t AddScalarPlugin::setTactic(int32_t tactic) noexcept
{
    WHERE_AM_I();
    mTactic = Tactic(tactic);
    return 0;
}

int32_t AddScalarPlugin::onShapeChange(PluginTensorDesc const *in, int32_t nbInputs, PluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t AddScalarPlugin::enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nElement = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        nElement *= inputDesc[0].dims.d[i];
    }
    dim3 grid(CEIL_DIVIDE(nElement, 256), 1, 1), block(256, 1, 1);

    switch (mTactic)
    {
    case Tactic::kNaive:
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Lag this branch to make the tactic "naïve" in this example
        addScalarKernel<<<grid, block, 0, stream>>>(reinterpret_cast<const float *>(inputs[0]), reinterpret_cast<float *>(outputs[0]), mScalar, nElement);
        break;
    case Tactic::kSmart: // Much smarter to call kernel directly, right?
        addScalarKernel<<<grid, block, 0, stream>>>(reinterpret_cast<const float *>(inputs[0]), reinterpret_cast<float *>(outputs[0]), mScalar, nElement);
        break;
    default:
        printf("should NOT be here\n");
        assert(0);
    }
    return 0;
}

IPluginV3 *AddScalarPlugin::attachToContext(IPluginResourceContext *context) noexcept
{
    WHERE_AM_I();
    return clone();
}

PluginFieldCollection const *AddScalarPlugin::getFieldsToSerialize() noexcept
{
    WHERE_AM_I();
    return &mFCToSerialize;
}

AddScalarPluginCreator::AddScalarPluginCreator()
{
    WHERE_AM_I();
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("scalar", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

char const *AddScalarPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

char const *AddScalarPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

PluginFieldCollection const *AddScalarPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &mFC;
}

IPluginV3 *AddScalarPluginCreator::createPlugin(char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept
{
    WHERE_AM_I();
    float scalar {0};
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        auto const fieldName(fc->fields[i].name);
        if (std::strcmp(fieldName, "scalar") == 0)
        {
            scalar = *static_cast<float const *>(fc->fields[i].data);
        }
    }
    return new AddScalarPlugin(scalar);
}

char const *AddScalarPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

REGISTER_TENSORRT_PLUGIN(AddScalarPluginCreator);

} // namespace nvinfer1
