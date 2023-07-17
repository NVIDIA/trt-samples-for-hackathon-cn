/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include <NvInfer.h>
#include <cstdio>
#include <cuda_fp16.h>
#include <string>
#include <vector>

#define WARP_SIZE           32
#define CEIL_DIVISION(X, Y) (((X) + (Y)-1) / (Y))
#define ALIGN_TO(X, Y)      (CEIL_DIVISION(X, Y) * (Y))

template<typename T>
__device__ inline T negtiveInfinity();

template<>
__device__ inline float negtiveInfinity<float>()
{
    return (float)-3.0e38;
}

template<>
__device__ inline half negtiveInfinity<half>()
{
    return (half)-6.0e4;
}

// +------- Debug wrapper ------------------------------------------------------
#if DEBUG_ENABLE
    #define DEBUG_FUNC()                                 \
        do                                               \
        {                                                \
            printf("[%s]: this=->%p\n", __func__, this); \
        } while (0);
#else
    #define DEBUG_FUNC()
#endif // DEBUG_ENABLE

// +------- Plguin -------------------------------------------------------------
namespace
{
static const char *PLUGIN_NAME {"MaskPlugin"};
static const char *PLUGIN_VERSION {"1"};
} // namespace

namespace nvinfer1
{
namespace plugin
{
// +------- Plugin body --------------------------------------------------------
class MaskPlugin : public IPluginV2DynamicExt
{
private:
    const std::string name_;
    std::string       namespace_;

public:
    MaskPlugin(const std::string &name):
        name_(name)
    {
        DEBUG_FUNC();
    }

    MaskPlugin(const std::string &name, const void *buffer, size_t length):
        name_(name)
    {
        DEBUG_FUNC();
    }

    MaskPlugin() = delete;

    ~MaskPlugin()
    {
        DEBUG_FUNC();
    }

    size_t getSerializationSize() const noexcept override
    {
        DEBUG_FUNC();
        return 0;
    }

    void serialize(void *buffer) const noexcept override
    {
        DEBUG_FUNC();
    }

    IPluginV2DynamicExt *clone() const noexcept override
    {
        DEBUG_FUNC();
        auto p = new MaskPlugin(name_);
        p->setPluginNamespace(namespace_.c_str());
        return p;
    }

    int getNbOutputs() const noexcept override
    {
        DEBUG_FUNC();
        return 3;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override
    {
        DEBUG_FUNC();
        DimsExprs out;
        switch (outputIndex)
        {
        case 0:
        case 1:
            out.nbDims = 4;
            out.d[0]   = inputs[0].d[0];
            out.d[1]   = exprBuilder.constant(4);
            out.d[2]   = inputs[0].d[1];
            out.d[3]   = inputs[0].d[1];
            break;
        case 2:
            out.nbDims = 3;
            out.d[0]   = inputs[0].d[0];
            out.d[1]   = inputs[0].d[1];
            out.d[2]   = exprBuilder.constant(320);
            break;
        default: // should NOT be here!
            out.nbDims = 0;
        }
        return out;
    }

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        DEBUG_FUNC();
        if (inOut[pos].format != TensorFormat::kLINEAR)
        {
            return false;
        }

        bool res = false;
        switch (pos)
        {
        case 0:
            res = (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF) && inOut[0].dims.nbDims == 3;
            break;
        case 1:
            res = inOut[1].type == DataType::kINT32 && inOut[1].dims.nbDims == 1;
            break;
        case 2:
        case 3:
            res = (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) && inOut[pos].dims.nbDims == 4;
            break;
        case 4:
            res = (inOut[4].type == DataType::kFLOAT || inOut[4].type == DataType::kHALF) && inOut[4].dims.nbDims == 3;
            break;
        default: // should NOT be here!
            break;
        }
#if DEBUG_ENABLE
        printf("Dim(");
        for (int i = 0; i < 2; ++i)
        {
            printf("%d,", inOut[i].dims.nbDims);
        }
        printf("),res(%d,%d),(%d,%d)\n", pos, int(res), int(inOut[0].type), int(inOut[1].type));
#endif
        return res;
    }

    DataType getOutputDataType(int outputIndex, const DataType *inputTypes, int nbInputs) const noexcept override
    {
        DEBUG_FUNC();
        return inputTypes[0];
    }

    void configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override
    {
        DEBUG_FUNC();
#if DEBUG_ENABLE
        printf("[MaskPlugin::configurePlugin]: maxBatchSize=%d, maxSequenceLength=%d\n", in[0].max.d[0], in[0].max.d[1]);
#endif
    }

    size_t getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override
    {
        DEBUG_FUNC();
        return 0;
    }

    void setPluginNamespace(const char *szNamespace) noexcept override
    {
        DEBUG_FUNC();
        namespace_ = szNamespace;
    }
    const char *getPluginNamespace() const noexcept override
    {
        DEBUG_FUNC();
        return namespace_.c_str();
    }
    const char *getPluginType() const noexcept override
    {
        DEBUG_FUNC();
        return PLUGIN_NAME;
    }
    const char *getPluginVersion() const noexcept override
    {
        DEBUG_FUNC();
        return PLUGIN_VERSION;
    }
    int initialize() noexcept override
    {
        DEBUG_FUNC();
        return 0;
    }
    void terminate() noexcept override
    {
        DEBUG_FUNC();
        return;
    }

    void destroy() noexcept override
    {
        DEBUG_FUNC();
        delete this;
    }

    int32_t enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
}; // class MaskPlugin

class MaskPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    MaskPluginCreator()
    {
        fc_.nbFields = attr_.size();
        fc_.fields   = attr_.data();
    }

    ~MaskPluginCreator() {}

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
    {
        DEBUG_FUNC();
        return new MaskPlugin(name);
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
    {
        return new MaskPlugin(name, serialData, serialLength);
    }

    void setPluginNamespace(const char *szNamespace) noexcept override
    {
        namespace_ = szNamespace;
    }

    const char *getPluginNamespace() const noexcept override
    {
        return namespace_.c_str();
    }

    const char *getPluginName() const noexcept override
    {
        return PLUGIN_NAME;
    }

    const char *getPluginVersion() const noexcept override
    {
        return PLUGIN_VERSION;
    }

    const PluginFieldCollection *getFieldNames() noexcept override
    {
        return &fc_;
    }
}; // class MaskPluginCreator

} // namespace plugin

} // namespace nvinfer1
