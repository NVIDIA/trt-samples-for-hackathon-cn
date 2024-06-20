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

#include "PushLeftPlugin.h"

#define N_BLOCK_SIZE 32

// kernel for GPU
// Stage1: get number of non-zero elements in each batch
__global__ void pushLeftStage1Kernel(float const *const pInput, int *const pWorkspace, int const nMaxSequenceLength, float const epsilon = 1.0e-5)
{
    int const src      = blockIdx.x * nMaxSequenceLength;
    int       nNonZero = 0;

    typedef cub::WarpReduce<int>                WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp;

    for (int i = threadIdx.x; i < nMaxSequenceLength; i += blockDim.x)
    {
        nNonZero += abs(pInput[src + i]) > epsilon;
    }
    __syncthreads();

    pWorkspace[blockIdx.x] = WarpReduce(temp).Reduce(nNonZero, cub::Sum());
}

// Stage2: get maximum of non-zero elements in each batch, i.e. shape of output
__global__ void pushLeftStage2Kernel(int const *const pWorkspace, int *const pOutput1, int const nBatchSize)
{
    typedef cub::WarpReduce<int>                WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp;
    pOutput1[0] = WarpReduce(temp).Reduce(pWorkspace[threadIdx.x], cub::Max());
}

// Stage3: fill non-zero elements to output buffer
__global__ void pushLeftStage3Kernel(float const *const input, float *const output0, int const *const output1, int const nMaxSequenceLength, float const epsilon = 1.0e-5)
{
    int const src = blockIdx.x * nMaxSequenceLength;
    int       dst = blockIdx.x * output1[0];
    for (int i = 0; i < nMaxSequenceLength; ++i)
    {
        if (abs(input[src + i]) > epsilon)
        {
            output0[dst] = input[src + i];
            ++dst;
        }
    }
}

namespace nvinfer1
{
PushLeftPlugin::PushLeftPlugin()
{
    WHERE_AM_I();
    initFieldsToSerialize();
}

void PushLeftPlugin::initFieldsToSerialize()
{
    WHERE_AM_I();
    mDataToSerialize.clear();
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields   = mDataToSerialize.data();
}

IPluginCapability *PushLeftPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3 *PushLeftPlugin::clone() noexcept
{
    WHERE_AM_I();
    std::unique_ptr<PushLeftPlugin> p {std::make_unique<PushLeftPlugin>(*this)};
    p->initFieldsToSerialize();
    return p.release();
}

char const *PushLeftPlugin::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

char const *PushLeftPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

char const *PushLeftPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

int32_t PushLeftPlugin::configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t PushLeftPlugin::getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    outputTypes[0] = inputTypes[0];
    outputTypes[1] = DataType::kINT32;
    return 0;
}

int32_t PushLeftPlugin::getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs, int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    outputs[0].nbDims = 2;
    outputs[0].d[0]   = inputs[0].d[0];

    auto maxValue   = inputs[0].d[1];
    auto optValue   = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *maxValue, *exprBuilder.constant(2));
    outputs[0].d[1] = exprBuilder.declareSizeTensor(1, *optValue, *maxValue);

    // We must have such an output size tensor (with dim == 0) to notify the shape of output tensor above
    outputs[1].nbDims = 0;
    return 0;
}

bool PushLeftPlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
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
    case 2:
        res = inOut[2].desc.type == DataType::kINT32 && inOut[2].desc.format == TensorFormat::kLINEAR;
        break;
    default: // should NOT be here!
        res = false;
    }
    PRINT_FORMAT_COMBINATION();
    return res;
}

int32_t PushLeftPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 2;
}

size_t PushLeftPlugin::getWorkspaceSize(DynamicPluginTensorDesc const *inputs, int32_t nbInputs, DynamicPluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return sizeof(int) * inputs[0].max.d[0];
}

int32_t PushLeftPlugin::getValidTactics(int32_t *tactics, int32_t nbTactics) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t PushLeftPlugin::getNbTactics() noexcept
{
    WHERE_AM_I();
    return 0;
}

char const *PushLeftPlugin::getTimingCacheID() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t PushLeftPlugin::getFormatCombinationLimit() noexcept
{
    WHERE_AM_I();
    return 1;
}

char const *PushLeftPlugin::getMetadataString() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t PushLeftPlugin::setTactic(int32_t tactic) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t PushLeftPlugin::onShapeChange(PluginTensorDesc const *in, int32_t nbInputs, PluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t PushLeftPlugin::enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nBatchSize         = inputDesc[0].dims.d[0];
    int nMaxSequenceLength = inputDesc[0].dims.d[1];

    const float *pInput     = reinterpret_cast<const float *>(inputs[0]);
    float       *pOutput0   = reinterpret_cast<float *>(outputs[0]);
    int         *pOutput1   = reinterpret_cast<int *>(outputs[1]);
    int         *pWorkspace = reinterpret_cast<int *>(workspace);

    pushLeftStage1Kernel<<<nBatchSize, N_BLOCK_SIZE, 0, stream>>>(pInput, pWorkspace, nMaxSequenceLength);

    pushLeftStage2Kernel<<<1, ALIGN_TO(nBatchSize, N_BLOCK_SIZE), 0, stream>>>(pWorkspace, pOutput1, nBatchSize);

    pushLeftStage3Kernel<<<nBatchSize, 1, 0, stream>>>(pInput, pOutput0, pOutput1, nMaxSequenceLength);

    return 0;
}

IPluginV3 *PushLeftPlugin::attachToContext(IPluginResourceContext *context) noexcept
{
    WHERE_AM_I();
    return clone();
}

PluginFieldCollection const *PushLeftPlugin::getFieldsToSerialize() noexcept
{
    WHERE_AM_I();
    return &mFCToSerialize;
}

PushLeftPluginCreator::PushLeftPluginCreator()
{
    WHERE_AM_I();
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

char const *PushLeftPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

char const *PushLeftPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

PluginFieldCollection const *PushLeftPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &mFC;
}

IPluginV3 *PushLeftPluginCreator::createPlugin(char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept
{
    WHERE_AM_I();
    return new PushLeftPlugin();
}

char const *PushLeftPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

REGISTER_TENSORRT_PLUGIN(PushLeftPluginCreator);

} // namespace nvinfer1
