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

#include "MyReshapePlugin.h"

namespace nvinfer1
{
MyReshapePlugin::MyReshapePlugin()
{
    WHERE_AM_I();
    initFieldsToSerialize();
}

void MyReshapePlugin::initFieldsToSerialize()
{
    WHERE_AM_I();
    mDataToSerialize.clear();
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields   = mDataToSerialize.data();
}

IPluginCapability *MyReshapePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3 *MyReshapePlugin::clone() noexcept
{
    WHERE_AM_I();
    std::unique_ptr<MyReshapePlugin> p {std::make_unique<MyReshapePlugin>(*this)};
    p->initFieldsToSerialize();
    return p.release();
}

char const *MyReshapePlugin::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

char const *MyReshapePlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

char const *MyReshapePlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

int32_t MyReshapePlugin::configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t MyReshapePlugin::getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t MyReshapePlugin::getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs, int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();

    printf("nbInputs=%d, nbShapeInputs=%d\n", nbInputs, nbShapeInputs);
    outputs[0].nbDims = shapeInputs[0].nbDims;
    outputs[0].d[0]   = shapeInputs[0].d[0]; // Set output shape by shape input tensor
    outputs[0].d[1]   = shapeInputs[0].d[1];
    outputs[0].d[2]   = shapeInputs[0].d[2];
    return 0;
}

bool MyReshapePlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res {false};
    switch (pos) // shape input tensor is not included
    {
    case 0:
        res = inOut[0].desc.type == DataType::kFLOAT && inOut[0].desc.format == TensorFormat::kLINEAR;
        break;
    case 1:
        res = inOut[pos].desc.type == inOut[0].desc.type && inOut[pos].desc.format == inOut[0].desc.format;
        break;
    default: // should NOT be here!
        res = false;
    }
    PRINT_FORMAT_COMBINATION();
    return res;
}

int32_t MyReshapePlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

size_t MyReshapePlugin::getWorkspaceSize(DynamicPluginTensorDesc const *inputs, int32_t nbInputs, DynamicPluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t MyReshapePlugin::getValidTactics(int32_t *tactics, int32_t nbTactics) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t MyReshapePlugin::getNbTactics() noexcept
{
    WHERE_AM_I();
    return 0;
}

char const *MyReshapePlugin::getTimingCacheID() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t MyReshapePlugin::getFormatCombinationLimit() noexcept
{
    WHERE_AM_I();
    return 1;
}

char const *MyReshapePlugin::getMetadataString() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t MyReshapePlugin::setTactic(int32_t tactic) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t MyReshapePlugin::onShapeChange(PluginTensorDesc const *in, int32_t nbInputs, PluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t MyReshapePlugin::enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nElement = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        nElement *= inputDesc[0].dims.d[i];
    }
    cudaMemcpyAsync(reinterpret_cast<float *>(outputs[0]), reinterpret_cast<const float *>(inputs[0]), sizeof(float) * nElement, cudaMemcpyDeviceToDevice, stream);
    return 0;
}

IPluginV3 *MyReshapePlugin::attachToContext(IPluginResourceContext *context) noexcept
{
    WHERE_AM_I();
    return clone();
}

PluginFieldCollection const *MyReshapePlugin::getFieldsToSerialize() noexcept
{
    WHERE_AM_I();
    return &mFCToSerialize;
}

MyReshapePluginCreator::MyReshapePluginCreator()
{
    WHERE_AM_I();
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

char const *MyReshapePluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

char const *MyReshapePluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

PluginFieldCollection const *MyReshapePluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &mFC;
}

IPluginV3 *MyReshapePluginCreator::createPlugin(char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept
{
    WHERE_AM_I();
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        auto const fieldName(fc->fields[i].name);
        if (std::strcmp(fieldName, "tensorrt_plugin_shape_input_indices") == 0)
        {
            printf("HAHAAHHHHHHHHHHHHAHAHHAHAHAHAAHAA\n");
            //scalar = *static_cast<float const *>(fc->fields[i].data);
        }
    }

    return new MyReshapePlugin();
}

char const *MyReshapePluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

REGISTER_TENSORRT_PLUGIN(MyReshapePluginCreator);

} // namespace nvinfer1
