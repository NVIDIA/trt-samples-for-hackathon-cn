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

#include "NcclSendRecvPlugin.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>

ThreadSafeLoggerFinder gLoggerFinder;

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder *finder)
{
    gLoggerFinder.setLoggerFinder(finder);
}

namespace nvinfer1
{

namespace
{
int32_t checkNccl(ncclResult_t const code, char const *call, char const *file, int32_t line)
{
    if (code == ncclSuccess)
    {
        return 0;
    }
    std::fprintf(stderr, "[NcclSendRecvPlugin] NCCL error at %s:%d %s -> %s\n", file, line, call, ncclGetErrorString(code));
    return -1;
}

#define NCCL_CHECK(X)                                    \
    do                                                   \
    {                                                    \
        if (checkNccl((X), #X, __FILE__, __LINE__) != 0) \
        {                                                \
            return -1;                                   \
        }                                                \
    } while (0)

} // namespace

NcclSendRecvPlugin::NcclSendRecvPlugin(
    int32_t                                         mode,
    int32_t                                         rank,
    int32_t                                         peer,
    int32_t                                         worldSize,
    std::array<int8_t, NCCL_UNIQUE_ID_BYTES> const &uniqueId):
    mMode(mode),
    mRank(rank), mPeer(peer), mWorldSize(worldSize), mUniqueId(uniqueId)
{
    WHERE_AM_I();
}

IPluginCapability *NcclSendRecvPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
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

IPluginV3 *NcclSendRecvPlugin::clone() noexcept
{
    std::unique_ptr<NcclSendRecvPlugin> p {std::make_unique<NcclSendRecvPlugin>(*this)};
    return p.release();
}

char const *NcclSendRecvPlugin::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

char const *NcclSendRecvPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

char const *NcclSendRecvPlugin::getPluginNamespace() const noexcept
{
    return PLUGIN_NAMESPACE;
}

int32_t NcclSendRecvPlugin::configurePlugin(
    DynamicPluginTensorDesc const *in,
    int32_t                        nbInputs,
    DynamicPluginTensorDesc const *out,
    int32_t                        nbOutputs) noexcept
{
    return 0;
}

int32_t NcclSendRecvPlugin::getOutputDataTypes(
    DataType       *outputTypes,
    int32_t         nbOutputs,
    DataType const *inputTypes,
    int32_t         nbInputs) const noexcept
{
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t NcclSendRecvPlugin::getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs, int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept
{
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < outputs[0].nbDims; ++i)
    {
        outputs[0].d[i] = inputs[0].d[i];
    }
    return 0;
}

bool NcclSendRecvPlugin::supportsFormatCombination(
    int32_t                        pos,
    DynamicPluginTensorDesc const *inOut,
    int32_t                        nbInputs,
    int32_t                        nbOutputs) noexcept
{
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
    return res;
}

int32_t NcclSendRecvPlugin::getNbOutputs() const noexcept
{
    return 1;
}

size_t NcclSendRecvPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const *inputs,
    int32_t                        nbInputs,
    DynamicPluginTensorDesc const *outputs,
    int32_t                        nbOutputs) const noexcept
{
    return 0;
}

int32_t NcclSendRecvPlugin::getValidTactics(int32_t *tactics, int32_t nbTactics) noexcept
{
    return 0;
}

int32_t NcclSendRecvPlugin::getNbTactics() noexcept
{
    return 0;
}

char const *NcclSendRecvPlugin::getTimingCacheID() noexcept
{
    return nullptr;
}

int32_t NcclSendRecvPlugin::getFormatCombinationLimit() noexcept
{
    return 1;
}

char const *NcclSendRecvPlugin::getMetadataString() noexcept
{
    return nullptr;
}

int32_t NcclSendRecvPlugin::setTactic(int32_t tactic) noexcept
{
    return 0;
}

int32_t NcclSendRecvPlugin::onShapeChange(
    PluginTensorDesc const *in,
    int32_t                 nbInputs,
    PluginTensorDesc const *out,
    int32_t                 nbOutputs) noexcept
{
    return 0;
}

int32_t NcclSendRecvPlugin::initializeCommIfNeeded() noexcept
{
    if (mInitialized)
    {
        return 0;
    }
    ncclUniqueId uniqueId {};
    std::memcpy(uniqueId.internal, mUniqueId.data(), NCCL_UNIQUE_ID_BYTES);
    NCCL_CHECK(ncclCommInitRank(&mComm, mWorldSize, uniqueId, mRank));
    mInitialized = true;
    return 0;
}

int32_t NcclSendRecvPlugin::enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    if (initializeCommIfNeeded() != 0)
    {
        return -1;
    }

    size_t nElement {1};
    for (int32_t i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        nElement *= static_cast<size_t>(inputDesc[0].dims.d[i]);
    }

    if (mMode == 0) // send
    {
        NCCL_CHECK(ncclSend(inputs[0], nElement, ncclFloat, mPeer, mComm, stream));
        CHECK(cudaMemcpyAsync(outputs[0], inputs[0], nElement * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        return 0;
    }

    // recv
    NCCL_CHECK(ncclRecv(outputs[0], nElement, ncclFloat, mPeer, mComm, stream));
    return 0;
}

IPluginV3 *NcclSendRecvPlugin::attachToContext(IPluginResourceContext *context) noexcept
{
    return clone();
}

PluginFieldCollection const *NcclSendRecvPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(PluginField("mode", &mMode, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("rank", &mRank, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("peer", &mPeer, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("world_size", &mWorldSize, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("unique_id", mUniqueId.data(), PluginFieldType::kINT8, NCCL_UNIQUE_ID_BYTES));
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields   = mDataToSerialize.data();
    return &mFCToSerialize;
}

NcclSendRecvPluginCreator::NcclSendRecvPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("mode", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("rank", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("peer", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("world_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("unique_id", nullptr, PluginFieldType::kINT8, NCCL_UNIQUE_ID_BYTES));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

char const *NcclSendRecvPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

char const *NcclSendRecvPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

PluginFieldCollection const *NcclSendRecvPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3 *NcclSendRecvPluginCreator::createPlugin(char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept
{
    int32_t                                  mode {0};
    int32_t                                  rank {0};
    int32_t                                  peer {1};
    int32_t                                  worldSize {2};
    std::array<int8_t, NCCL_UNIQUE_ID_BYTES> uniqueId {};

    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        auto const fieldName(fc->fields[i].name);
        if (std::strcmp(fieldName, "mode") == 0)
        {
            mode = *static_cast<int32_t const *>(fc->fields[i].data);
        }
        else if (std::strcmp(fieldName, "rank") == 0)
        {
            rank = *static_cast<int32_t const *>(fc->fields[i].data);
        }
        else if (std::strcmp(fieldName, "peer") == 0)
        {
            peer = *static_cast<int32_t const *>(fc->fields[i].data);
        }
        else if (std::strcmp(fieldName, "world_size") == 0)
        {
            worldSize = *static_cast<int32_t const *>(fc->fields[i].data);
        }
        else if (std::strcmp(fieldName, "unique_id") == 0)
        {
            int32_t const n = std::min(fc->fields[i].length, static_cast<int32_t>(NCCL_UNIQUE_ID_BYTES));
            std::memcpy(uniqueId.data(), fc->fields[i].data, static_cast<size_t>(n));
        }
    }
    return new NcclSendRecvPlugin(mode, rank, peer, worldSize, uniqueId);
}

char const *NcclSendRecvPluginCreator::getPluginNamespace() const noexcept
{
    return PLUGIN_NAMESPACE;
}

} // namespace nvinfer1

extern "C" nvinfer1::IPluginCreatorV3One *const *getCreators(int32_t &nbCreators)
{
    nbCreators = 1;
    static nvinfer1::NcclSendRecvPluginCreator  creator;
    static nvinfer1::IPluginCreatorV3One *const pluginCreatorList[] = {&creator};
    return pluginCreatorList;
}
