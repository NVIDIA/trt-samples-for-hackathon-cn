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

#include "AddScalarPlugin.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>

namespace
{

// Optional utility class to use TensorRT's Logger in plugin
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

    void logHello() noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mLoggerFinder != nullptr)
        {
            auto *logger = mLoggerFinder->findLogger();
            if (logger != nullptr)
            {
                // Logger exists, use TensorRT's logger to print hello world
                logger->log(nvinfer1::ILogger::Severity::kINFO, "\n\nHello\n");
                return;
            }
        }
        // Otherwise, fallback to standard output
        std::fprintf(stdout, "\nHello (fallback)\n\n");
        std::fflush(stdout);
    }
};

ThreadSafeLoggerFinder gLoggerFinder;

} // namespace

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder *finder)
{
    gLoggerFinder.setLoggerFinder(finder);
    gLoggerFinder.logHello(); // Log Hello during initialization, just for showing usage of log
}

// kernel for GPU
__global__ void addScalarKernel(float const *__restrict__ input, float *__restrict__ output, float const scalar, int32_t const nElement)
{
    int32_t const index = blockIdx.x * blockDim.x + threadIdx.x;
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
}

IPluginCapability *AddScalarPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3 *AddScalarPlugin::clone() noexcept
{
    WHERE_AM_I();
    try
    {
        std::unique_ptr<AddScalarPlugin> p {std::make_unique<AddScalarPlugin>(*this)};
        return p.release();
    }
    catch (...)
    {
        return nullptr;
    }
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
    if (in == nullptr || out == nullptr || nbInputs != 1 || nbOutputs != 1)
    {
        return -1;
    }
    return 0;
}

int32_t AddScalarPlugin::getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != 1 || nbOutputs != 1)
    {
        return -1;
    }
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t AddScalarPlugin::getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs, int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    if (inputs == nullptr || outputs == nullptr || nbInputs != 1 || nbOutputs != 1)
    {
        return -1;
    }
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
    default: // should NOT be here!
        break;
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
    return 0;
}

int32_t AddScalarPlugin::getNbTactics() noexcept
{
    WHERE_AM_I();
    return 0;
}

char const *AddScalarPlugin::getTimingCacheID() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t AddScalarPlugin::getFormatCombinationLimit() noexcept
{
    WHERE_AM_I();
    return 0;
}

char const *AddScalarPlugin::getMetadataString() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t AddScalarPlugin::setTactic(int32_t tactic) noexcept
{
    WHERE_AM_I();
    if (tactic != 0)
    {
        return -1;
    }
    return 0;
}

int32_t AddScalarPlugin::onShapeChange(PluginTensorDesc const *in, int32_t nbInputs, PluginTensorDesc const *out, int32_t nbOutputs) noexcept
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

int32_t AddScalarPlugin::enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    gLoggerFinder.logHello(); // Log Hello during enqueue, just for showiing usage of log
    if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr)
    {
        return -1;
    }
    if (inputDesc[0].type != DataType::kFLOAT || outputDesc[0].type != DataType::kFLOAT)
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
    dim3          grid(CEIL_DIVIDE(nElement, 256), 1, 1), block(256, 1, 1);
    addScalarKernel<<<grid, block, 0, stream>>>(reinterpret_cast<float const *>(inputs[0]), reinterpret_cast<float *>(outputs[0]), mScalar, nElement);

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

IPluginV3 *AddScalarPlugin::attachToContext(IPluginResourceContext *context) noexcept
{
    WHERE_AM_I();
    try
    {
        return clone();
    }
    catch (...)
    {
        return nullptr;
    }
}

PluginFieldCollection const *AddScalarPlugin::getFieldsToSerialize() noexcept
{
    WHERE_AM_I();
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back(PluginField("scalar", &mScalar, PluginFieldType::kFLOAT32, 1));
        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields   = mDataToSerialize.data();
        return &mFCToSerialize;
    }
    catch (...)
    {
        return nullptr;
    }
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
    try
    {
        float scalar {0.f};

        if (fc != nullptr && fc->fields != nullptr)
        {
            for (int32_t i = 0; i < fc->nbFields; ++i)
            {
                auto const fieldName(fc->fields[i].name);
                if (fieldName != nullptr && std::strcmp(fieldName, "scalar") == 0)
                {
                    if (fc->fields[i].data == nullptr || fc->fields[i].type != PluginFieldType::kFLOAT32 || fc->fields[i].length != 1)
                    {
                        return nullptr;
                    }
                    scalar = *static_cast<float const *>(fc->fields[i].data);
                }
            }
        }

        return new AddScalarPlugin(scalar);
    }
    catch (...)
    {
        return nullptr;
    }
}

char const *AddScalarPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

} // namespace nvinfer1

extern "C" nvinfer1::IPluginCreatorV3One *const *getCreators(int32_t &nbCreators)
{
    nbCreators = 1;
    static nvinfer1::AddScalarPluginCreator     creator;
    static nvinfer1::IPluginCreatorV3One *const pluginCreatorList[] = {&creator};
    return pluginCreatorList;
}
