/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

#include "CuFFTPlugin.h"

ThreadSafeLoggerFinder gLoggerFinder;

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder *finder)
{
    gLoggerFinder.setLoggerFinder(finder);
}

namespace nvinfer1
{
// class CuFFTPlugin
CuFFTPlugin::CuFFTPlugin(int32_t const mode, int32_t const signalLength, int32_t const inverse):
    mMode(mode), mSignalLength(signalLength), mInverse(inverse)
{
    WHERE_AM_I();
}

CuFFTPlugin::CuFFTPlugin(CuFFTPlugin const &p)
{
    WHERE_AM_I();
    mMode         = p.mMode;
    mSignalLength = p.mSignalLength;
    mInverse      = p.mInverse;
    // The cuFFT plan is deliberately NOT copied: it belongs to the context that created it.
    mPlan      = 0;
    mPlanBatch = 0;
}

CuFFTPlugin::~CuFFTPlugin()
{
    WHERE_AM_I();
    destroyPlan();
}

void CuFFTPlugin::destroyPlan()
{
    WHERE_AM_I();
    if (mPlan != 0)
    {
        cufftDestroy(mPlan);
        mPlan      = 0;
        mPlanBatch = 0;
    }
}

IPluginCapability *CuFFTPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

CuFFTPlugin *CuFFTPlugin::clone() noexcept
{
    WHERE_AM_I();
    std::unique_ptr<CuFFTPlugin> p {std::make_unique<CuFFTPlugin>(*this)};
    return p.release();
}

char const *CuFFTPlugin::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

char const *CuFFTPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

char const *CuFFTPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

int32_t CuFFTPlugin::configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CuFFTPlugin::getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    outputTypes[0] = inputTypes[0]; // Always float32; complex is carried as a trailing dimension of 2
    return 0;
}

int32_t CuFFTPlugin::getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs, int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    // ONNX has no complex type, so a complex tensor is represented as a real tensor with a
    // trailing dimension of 2 (real, imaginary) -- the same convention the ONNX `DFT`
    // operator uses. The three modes therefore change the rank as well as the extent.
    switch (static_cast<FFTMode>(mMode))
    {
    case FFTMode::kC2C: // [.., n, 2] -> [.., n, 2]
        outputs[0] = inputs[0];
        break;
    case FFTMode::kR2C: // [.., n] -> [.., n / 2 + 1, 2]
        outputs[0].nbDims = inputs[0].nbDims + 1;
        for (int32_t i = 0; i < inputs[0].nbDims - 1; ++i)
        {
            outputs[0].d[i] = inputs[0].d[i];
        }
        outputs[0].d[outputs[0].nbDims - 2] = exprBuilder.constant(mSignalLength / 2 + 1);
        outputs[0].d[outputs[0].nbDims - 1] = exprBuilder.constant(2);
        break;
    case FFTMode::kC2R: // [.., n / 2 + 1, 2] -> [.., n]
        outputs[0].nbDims = inputs[0].nbDims - 1;
        for (int32_t i = 0; i < outputs[0].nbDims - 1; ++i)
        {
            outputs[0].d[i] = inputs[0].d[i];
        }
        outputs[0].d[outputs[0].nbDims - 1] = exprBuilder.constant(mSignalLength);
        break;
    }
    return 0;
}

bool CuFFTPlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res {false};
    switch (pos)
    {
    case 0:
        // cuFFT's single-precision entry points take float; there is no half-precision path here
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

int32_t CuFFTPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

size_t CuFFTPlugin::getWorkspaceSize(DynamicPluginTensorDesc const *inputs, int32_t nbInputs, DynamicPluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0; // cuFFT manages its own work area inside the plan
}

int32_t CuFFTPlugin::getValidTactics(int32_t *tactics, int32_t nbTactics) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CuFFTPlugin::getNbTactics() noexcept
{
    WHERE_AM_I();
    return 0;
}

char const *CuFFTPlugin::getTimingCacheID() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t CuFFTPlugin::getFormatCombinationLimit() noexcept
{
    WHERE_AM_I();
    return 1;
}

char const *CuFFTPlugin::getMetadataString() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t CuFFTPlugin::setTactic(int32_t tactic) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CuFFTPlugin::onShapeChange(PluginTensorDesc const *in, int32_t nbInputs, PluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    // A cuFFT plan is tied to (transform type, length, batch), and the batch here comes from
    // the leading dimensions, which are dynamic. `onShapeChange` is the right place to notice
    // that: it is called whenever the runtime shapes change, so the plan is rebuilt only then
    // rather than on every `enqueue`.
    int32_t       batch {1};
    int32_t const nbBatchDims = (static_cast<FFTMode>(mMode) == FFTMode::kC2R) ? in[0].dims.nbDims - 2 : ((static_cast<FFTMode>(mMode) == FFTMode::kC2C) ? in[0].dims.nbDims - 2 : in[0].dims.nbDims - 1);
    for (int32_t i = 0; i < nbBatchDims; ++i)
    {
        batch *= in[0].dims.d[i];
    }
    if (batch == mPlanBatch && mPlan != 0)
    {
        return 0;
    }
    destroyPlan();

    cufftType type {CUFFT_C2C};
    switch (static_cast<FFTMode>(mMode))
    {
    case FFTMode::kC2C:
        type = CUFFT_C2C;
        break;
    case FFTMode::kR2C:
        type = CUFFT_R2C;
        break;
    case FFTMode::kC2R:
        type = CUFFT_C2R;
        break;
    }
    if (!CHECK_CUFFT(cufftPlan1d(&mPlan, mSignalLength, type, batch)))
    {
        return 1;
    }
    mPlanBatch = batch;
    return 0;
}

int32_t CuFFTPlugin::enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    if (mPlan == 0)
    {
        std::cerr << "CuFFTPlugin: no plan, onShapeChange was not called" << std::endl;
        return 1;
    }
    // Binding the plan to TensorRT's stream is what keeps the plugin asynchronous. Forgetting
    // this call is the classic cuFFT-in-a-plugin bug: it still produces correct numbers,
    // because cuFFT then uses the default stream and the driver serialises everything, but
    // the plugin silently becomes a synchronisation point in the middle of the engine.
    if (!CHECK_CUFFT(cufftSetStream(mPlan, stream)))
    {
        return 1;
    }

    switch (static_cast<FFTMode>(mMode))
    {
    case FFTMode::kC2C:
        if (!CHECK_CUFFT(cufftExecC2C(mPlan, (cufftComplex *)inputs[0], (cufftComplex *)outputs[0], mInverse ? CUFFT_INVERSE : CUFFT_FORWARD)))
        {
            return 1;
        }
        break;
    case FFTMode::kR2C:
        if (!CHECK_CUFFT(cufftExecR2C(mPlan, (cufftReal *)inputs[0], (cufftComplex *)outputs[0])))
        {
            return 1;
        }
        break;
    case FFTMode::kC2R:
        if (!CHECK_CUFFT(cufftExecC2R(mPlan, (cufftComplex *)inputs[0], (cufftReal *)outputs[0])))
        {
            return 1;
        }
        break;
    }
    return 0;
}

IPluginV3 *CuFFTPlugin::attachToContext(IPluginResourceContext *context) noexcept
{
    WHERE_AM_I();
    // The clone starts without a plan; `onShapeChange` builds one for the real batch size
    return this->clone();
}

PluginFieldCollection const *CuFFTPlugin::getFieldsToSerialize() noexcept
{
    WHERE_AM_I();
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(PluginField("mode", &mMode, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("signal_length", &mSignalLength, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("inverse", &mInverse, PluginFieldType::kINT32, 1));
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields   = mDataToSerialize.data();
    return &mFCToSerialize;
}

CuFFTPluginCreator::CuFFTPluginCreator()
{
    WHERE_AM_I();
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("mode", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("signal_length", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("inverse", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

char const *CuFFTPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

char const *CuFFTPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

PluginFieldCollection const *CuFFTPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &mFC;
}

IPluginV3 *CuFFTPluginCreator::createPlugin(char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept
{
    WHERE_AM_I();
    int32_t mode {0}, signalLength {0}, inverse {0};
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        std::string const field_name(fc->fields[i].name);
        if (field_name.compare("mode") == 0)
        {
            mode = *reinterpret_cast<int32_t const *>(fc->fields[i].data);
        }
        else if (field_name.compare("signal_length") == 0)
        {
            signalLength = *reinterpret_cast<int32_t const *>(fc->fields[i].data);
        }
        else if (field_name.compare("inverse") == 0)
        {
            inverse = *reinterpret_cast<int32_t const *>(fc->fields[i].data);
        }
    }
    return new CuFFTPlugin(mode, signalLength, inverse);
}

char const *CuFFTPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAMESPACE;
}

} // namespace nvinfer1

extern "C" nvinfer1::IPluginCreatorV3One *const *getCreators(int32_t &nbCreators)
{
    nbCreators = 1;
    static nvinfer1::CuFFTPluginCreator         creator;
    static nvinfer1::IPluginCreatorV3One *const pluginCreatorList[] = {&creator};
    return pluginCreatorList;
}
