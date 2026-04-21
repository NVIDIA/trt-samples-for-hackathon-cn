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

#include "NCCLAllReducePlugin.h"

#include <array>

namespace
{
inline int32_t checkNccl(ncclResult_t e)
{
    if (e != ncclSuccess)
    {
        std::cout << "NCCL error: " << ncclGetErrorString(e) << std::endl;
        return 1;
    }
    return 0;
}
} // namespace

namespace nvinfer1
{

NCCLAllReducePlugin::NCCLAllReducePlugin(int32_t rank, int32_t worldSize, ncclUniqueId const &uniqueId):
    mRank(rank),
    mWorldSize(worldSize),
    mUniqueId(uniqueId)
{
    WHERE_AM_I();
}

NCCLAllReducePlugin::~NCCLAllReducePlugin()
{
    if (mCommInited)
    {
        ncclCommDestroy(mComm);
        mCommInited = false;
    }
}

IPluginCapability *NCCLAllReducePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3 *NCCLAllReducePlugin::clone() noexcept
{
    WHERE_AM_I();
    std::unique_ptr<NCCLAllReducePlugin> p {std::make_unique<NCCLAllReducePlugin>(*this)};
    return p.release();
}

char const *NCCLAllReducePlugin::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

char const *NCCLAllReducePlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

char const *NCCLAllReducePlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

int32_t NCCLAllReducePlugin::configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t NCCLAllReducePlugin::getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t NCCLAllReducePlugin::getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs, int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < outputs[0].nbDims; ++i)
    {
        outputs[0].d[i] = inputs[0].d[i];
    }
    return 0;
}

bool NCCLAllReducePlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
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
    default:
        res = false;
    }
    PRINT_FORMAT_COMBINATION();
    return res;
}

int32_t NCCLAllReducePlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

size_t NCCLAllReducePlugin::getWorkspaceSize(DynamicPluginTensorDesc const *inputs, int32_t nbInputs, DynamicPluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t NCCLAllReducePlugin::getValidTactics(int32_t *tactics, int32_t nbTactics) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t NCCLAllReducePlugin::getNbTactics() noexcept
{
    WHERE_AM_I();
    return 0;
}

char const *NCCLAllReducePlugin::getTimingCacheID() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t NCCLAllReducePlugin::getFormatCombinationLimit() noexcept
{
    WHERE_AM_I();
    return 1;
}

char const *NCCLAllReducePlugin::getMetadataString() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t NCCLAllReducePlugin::setTactic(int32_t tactic) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t NCCLAllReducePlugin::onShapeChange(PluginTensorDesc const *in, int32_t nbInputs, PluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t NCCLAllReducePlugin::enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    if (!mCommInited)
    {
        if (checkNccl(ncclCommInitRank(&mComm, mWorldSize, mUniqueId, mRank)) != 0)
        {
            return 1;
        }
        mCommInited = true;
    }

    int64_t count {1};
    for (int32_t i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        count *= inputDesc[0].dims.d[i];
    }

    if (checkNccl(ncclAllReduce(inputs[0], outputs[0], count, ncclFloat, ncclSum, mComm, stream)) != 0)
    {
        return 1;
    }
    return 0;
}

IPluginV3 *NCCLAllReducePlugin::attachToContext(IPluginResourceContext *context) noexcept
{
    WHERE_AM_I();
    return clone();
}

PluginFieldCollection const *NCCLAllReducePlugin::getFieldsToSerialize() noexcept
{
    WHERE_AM_I();
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(PluginField("rank", &mRank, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("world_size", &mWorldSize, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("nccl_uid", mUniqueId.internal, PluginFieldType::kCHAR, sizeof(mUniqueId.internal)));
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields   = mDataToSerialize.data();
    return &mFCToSerialize;
}

NCCLAllReducePluginCreator::NCCLAllReducePluginCreator()
{
    WHERE_AM_I();
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("rank", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("world_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("nccl_uid", nullptr, PluginFieldType::kCHAR, sizeof(ncclUniqueId)));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

char const *NCCLAllReducePluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

char const *NCCLAllReducePluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

PluginFieldCollection const *NCCLAllReducePluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &mFC;
}

IPluginV3 *NCCLAllReducePluginCreator::createPlugin(char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept
{
    WHERE_AM_I();
    int32_t      rank {0};
    int32_t      worldSize {1};
    ncclUniqueId uniqueId {};

    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        auto const fieldName(fc->fields[i].name);
        if (std::strcmp(fieldName, "rank") == 0)
        {
            rank = *static_cast<int32_t const *>(fc->fields[i].data);
        }
        else if (std::strcmp(fieldName, "world_size") == 0)
        {
            worldSize = *static_cast<int32_t const *>(fc->fields[i].data);
        }
        else if (std::strcmp(fieldName, "nccl_uid") == 0)
        {
            std::memcpy(uniqueId.internal, fc->fields[i].data, sizeof(uniqueId.internal));
        }
    }

    return new NCCLAllReducePlugin(rank, worldSize, uniqueId);
}

char const *NCCLAllReducePluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

REGISTER_TENSORRT_PLUGIN(NCCLAllReducePluginCreator);

} // namespace nvinfer1
