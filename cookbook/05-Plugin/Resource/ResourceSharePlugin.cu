/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
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

#include "ResourceSharePlugin.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>

namespace
{

class ThreadSafeLoggerFinder
{
private:
    nvinfer1::ILoggerFinder *mLoggerFinder {nullptr};
    std::mutex               mMutex;

public:
    void setLoggerFinder(nvinfer1::ILoggerFinder *finder)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mLoggerFinder == nullptr && finder != nullptr)
        {
            mLoggerFinder = finder;
        }
    }
};

ThreadSafeLoggerFinder gLoggerFinder;

} // namespace

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder *finder)
{
    gLoggerFinder.setLoggerFinder(finder);
}

namespace nvinfer1
{

IPluginResource *SharedTextResource::clone() noexcept
{
    try
    {
        std::unique_ptr<SharedTextResource> resource {std::make_unique<SharedTextResource>()};
        resource->mLastSeed.store(mLastSeed.load());
        resource->mWriteCount.store(mWriteCount.load());
        return resource.release();
    }
    catch (...)
    {
        return nullptr;
    }
}

int32_t SharedTextResource::release() noexcept
{
    return 0;
}

ResourceSharePlugin::ResourceSharePlugin(bool const isWriter, int32_t const seed):
    mIsWriter(isWriter),
    mSeed(seed)
{
    WHERE_AM_I();
}

ResourceSharePlugin::ResourceSharePlugin(ResourceSharePlugin const &plugin):
    mIsWriter(plugin.mIsWriter),
    mSeed(plugin.mSeed)
{
    WHERE_AM_I();
}

ResourceSharePlugin::~ResourceSharePlugin() noexcept
{
    releaseSharedResource();
}

bool ResourceSharePlugin::acquireSharedResource() noexcept
{
    if (mHasAcquiredResource)
    {
        return mResource != nullptr;
    }

    auto *pluginRegistry = getPluginRegistry();
    if (pluginRegistry == nullptr)
    {
        return false;
    }

    SharedTextResource resourceTemplate;
    resourceTemplate.mLastSeed.store(mSeed);
    auto *resource = pluginRegistry->acquirePluginResource(RESOURCE_KEY, &resourceTemplate);
    if (resource == nullptr)
    {
        return false;
    }

    mResource            = static_cast<SharedTextResource *>(resource);
    mHasAcquiredResource = true;
    return true;
}

void ResourceSharePlugin::releaseSharedResource() noexcept
{
    if (!mHasAcquiredResource)
    {
        return;
    }
    mHasAcquiredResource = false;
    mResource            = nullptr;

    auto *pluginRegistry = getPluginRegistry();
    if (pluginRegistry != nullptr)
    {
        (void)pluginRegistry->releasePluginResource(RESOURCE_KEY);
    }
}

IPluginCapability *ResourceSharePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3 *ResourceSharePlugin::clone() noexcept
{
    WHERE_AM_I();
    try
    {
        std::unique_ptr<ResourceSharePlugin> plugin {std::make_unique<ResourceSharePlugin>(*this)};
        return plugin.release();
    }
    catch (...)
    {
        return nullptr;
    }
}

char const *ResourceSharePlugin::getPluginName() const noexcept
{
    WHERE_AM_I();
    return mIsWriter ? WRITER_PLUGIN_NAME : READER_PLUGIN_NAME;
}

char const *ResourceSharePlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

char const *ResourceSharePlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

int32_t ResourceSharePlugin::configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    if (in == nullptr || out == nullptr || nbInputs != 1 || nbOutputs != 1)
    {
        return -1;
    }
    return 0;
}

int32_t ResourceSharePlugin::getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != 1 || nbOutputs != 1)
    {
        return -1;
    }
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t ResourceSharePlugin::getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs, int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    if (inputs == nullptr || outputs == nullptr || nbInputs != 1 || nbOutputs != 1)
    {
        return -1;
    }
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < outputs[0].nbDims; ++i)
    {
        outputs[0].d[i] = inputs[0].d[i];
    }
    return 0;
}

bool ResourceSharePlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    if (inOut == nullptr || nbInputs != 1 || nbOutputs != 1 || pos < 0 || pos >= nbInputs + nbOutputs)
    {
        return false;
    }

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
        break;
    }
    PRINT_FORMAT_COMBINATION();
    return res;
}

int32_t ResourceSharePlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

size_t ResourceSharePlugin::getWorkspaceSize(DynamicPluginTensorDesc const *inputs, int32_t nbInputs, DynamicPluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t ResourceSharePlugin::getValidTactics(int32_t *tactics, int32_t nbTactics) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t ResourceSharePlugin::getNbTactics() noexcept
{
    WHERE_AM_I();
    return 0;
}

char const *ResourceSharePlugin::getTimingCacheID() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t ResourceSharePlugin::getFormatCombinationLimit() noexcept
{
    WHERE_AM_I();
    return 0;
}

char const *ResourceSharePlugin::getMetadataString() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t ResourceSharePlugin::setTactic(int32_t tactic) noexcept
{
    WHERE_AM_I();
    if (tactic != 0)
    {
        return -1;
    }
    return 0;
}

int32_t ResourceSharePlugin::onShapeChange(PluginTensorDesc const *in, int32_t nbInputs, PluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    if (in == nullptr || out == nullptr || nbInputs != 1 || nbOutputs != 1)
    {
        return -1;
    }
    for (int32_t i = 0; i < in[0].dims.nbDims; ++i)
    {
        if (in[0].dims.d[i] <= 0)
        {
            return -1;
        }
    }
    return 0;
}

int32_t ResourceSharePlugin::enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr)
    {
        return -1;
    }
    if (inputDesc[0].type != DataType::kFLOAT || outputDesc[0].type != DataType::kFLOAT)
    {
        return -1;
    }
    if (mResource == nullptr)
    {
        return -1;
    }

    int64_t nElement64 {1};
    for (int32_t i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        int32_t const dim = inputDesc[0].dims.d[i];
        nElement64 *= dim;
        if (dim <= 0 || nElement64 > std::numeric_limits<int32_t>::max())
        {
            return -1;
        }
    }
    int32_t const nElement = static_cast<int32_t>(nElement64);

    if (mIsWriter)
    {
        mResource->mLastSeed.store(mSeed);
        int32_t const writeCount = mResource->mWriteCount.fetch_add(1) + 1;
        std::fprintf(stdout, "[ResourceWriter] write shared resource: seed=%d, nElement=%d, writeCount=%d\n", mSeed, nElement, writeCount);
        std::fflush(stdout);
    }
    else
    {
        int32_t const seed       = mResource->mLastSeed.load();
        int32_t const writeCount = mResource->mWriteCount.load();
        std::fprintf(stdout, "[ResourceReader] read shared resource: seed=%d, writeCount=%d\n", seed, writeCount);
        std::fflush(stdout);
    }

    size_t const nByte = static_cast<size_t>(nElement) * sizeof(float);
    return (cudaMemcpyAsync(outputs[0], inputs[0], nByte, cudaMemcpyDeviceToDevice, stream) == cudaSuccess) ? 0 : -1;
}

IPluginV3 *ResourceSharePlugin::attachToContext(IPluginResourceContext *context) noexcept
{
    WHERE_AM_I();
    try
    {
        std::unique_ptr<ResourceSharePlugin> plugin {std::make_unique<ResourceSharePlugin>(*this)};
        if (!plugin->acquireSharedResource())
        {
            return nullptr;
        }
        return plugin.release();
    }
    catch (...)
    {
        return nullptr;
    }
}

PluginFieldCollection const *ResourceSharePlugin::getFieldsToSerialize() noexcept
{
    WHERE_AM_I();
    try
    {
        mDataToSerialize.clear();
        if (mIsWriter)
        {
            mDataToSerialize.emplace_back(PluginField("seed", &mSeed, PluginFieldType::kINT32, 1));
        }
        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields   = mDataToSerialize.data();
        return &mFCToSerialize;
    }
    catch (...)
    {
        return nullptr;
    }
}

ResourceWriterPluginCreator::ResourceWriterPluginCreator()
{
    WHERE_AM_I();
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("seed", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

char const *ResourceWriterPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return WRITER_PLUGIN_NAME;
}

char const *ResourceWriterPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

PluginFieldCollection const *ResourceWriterPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &mFC;
}

IPluginV3 *ResourceWriterPluginCreator::createPlugin(char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept
{
    WHERE_AM_I();
    try
    {
        int32_t seed {0};
        if (fc != nullptr && fc->fields != nullptr)
        {
            for (int32_t i = 0; i < fc->nbFields; ++i)
            {
                auto const fieldName(fc->fields[i].name);
                if (fieldName != nullptr && std::strcmp(fieldName, "seed") == 0)
                {
                    if (fc->fields[i].data == nullptr || fc->fields[i].type != PluginFieldType::kINT32 || fc->fields[i].length != 1)
                    {
                        return nullptr;
                    }
                    seed = *static_cast<int32_t const *>(fc->fields[i].data);
                }
            }
        }
        return new ResourceSharePlugin(true, seed);
    }
    catch (...)
    {
        return nullptr;
    }
}

char const *ResourceWriterPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

ResourceReaderPluginCreator::ResourceReaderPluginCreator()
{
    WHERE_AM_I();
    mFC.nbFields = 0;
    mFC.fields   = nullptr;
}

char const *ResourceReaderPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return READER_PLUGIN_NAME;
}

char const *ResourceReaderPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

PluginFieldCollection const *ResourceReaderPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &mFC;
}

IPluginV3 *ResourceReaderPluginCreator::createPlugin(char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept
{
    WHERE_AM_I();
    try
    {
        return new ResourceSharePlugin(false, 0);
    }
    catch (...)
    {
        return nullptr;
    }
}

char const *ResourceReaderPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

} // namespace nvinfer1

extern "C" nvinfer1::IPluginCreatorV3One *const *getCreators(int32_t &nbCreators)
{
    nbCreators = 2;
    static nvinfer1::ResourceWriterPluginCreator writerCreator;
    static nvinfer1::ResourceReaderPluginCreator readerCreator;
    static nvinfer1::IPluginCreatorV3One *const  pluginCreatorList[] = {&writerCreator, &readerCreator};
    return pluginCreatorList;
}
