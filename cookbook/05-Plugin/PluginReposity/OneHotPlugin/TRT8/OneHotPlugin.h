/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "cuda_fp16.h"

#include <NvInfer.h>
#include <cassert>
#include <iostream>
#include <vector>

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
#define CEIL(x, y)  (((x) + (y)-1) / (y))

namespace nvinfer1
{
namespace plugin
{
class OneHotPlugin : public IPluginV2DynamicExt
{
private:
    struct
    {
        int nEmbed;
        int nRow;
        int isFp16;
    } m;
    const char *mPluginNamespace;
    std::string mNamespace;

public:
    OneHotPlugin(int nEmbed, int isFp16)
    {
        m.nEmbed = nEmbed;
        m.isFp16 = isFp16;
    }

    OneHotPlugin() = delete;

    OneHotPlugin(const void *buffer, size_t length)
    {
        memcpy(&m, buffer, sizeof(m));
    }

    virtual size_t getSerializationSize() const noexcept override
    {
        return sizeof(m);
    }

    virtual void serialize(void *buffer) const noexcept override
    {
        memcpy(buffer, &m, sizeof(m));
    }

    IPluginV2DynamicExt *clone() const noexcept override
    {
        return new OneHotPlugin(&m, sizeof(m));
    }

    int getNbOutputs() const noexcept override
    {
        return 1;
    }

    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept override
    {
        return (m.isFp16 ? DataType::kHALF : DataType::kFLOAT);
    }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder) noexcept override
    {
        DimsExprs output(inputs[0]);
        output.nbDims += 1;
        output.d[output.nbDims - 1] = exprBuilder.constant(m.nEmbed);
        return output;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) noexcept override
    {
        if (inOut[pos].format != TensorFormat::kLINEAR)
            return false;
        if (pos == 0)
            return inOut[0].type == DataType::kINT32;
        else
            return inOut[1].type == (m.isFp16 ? DataType::kHALF : DataType::kFLOAT);
    }

    size_t getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const noexcept override
    {
        return 0;
    }

    int enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

    int initialize() noexcept override
    {
        return 0;
    }
    void terminate() noexcept override {}
    void destroy() noexcept override
    {
        delete this;
    }
    void setPluginNamespace(const char *szNamespace) noexcept override
    {
        mNamespace = szNamespace;
    }
    const char *getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }
    const char *getPluginType() const noexcept override
    {
        return "OneHotPlugin";
    }
    const char *getPluginVersion() const noexcept override
    {
        return "1";
    }

    void configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs) noexcept override
    {
        int nRow = 1;
        for (int i = 1; i < in[0].desc.dims.nbDims; i++)
            nRow *= in[0].desc.dims.d[i];
        m.nRow = nRow;
    }
};

class OneHotPluginCreator : public IPluginCreator
{
private:
    std::string                     mNamespace;
    static PluginFieldCollection    mFC;
    static std::vector<PluginField> mPluginAttributes;

public:
    OneHotPluginCreator()
    {
        mPluginAttributes.emplace_back(PluginField("nEmbed", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("isFp16", nullptr, PluginFieldType::kINT32, 0));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields   = mPluginAttributes.data();
    }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
    {
        int nEmbed = 1, isFp16 = 0;
        for (int i = 0; i < fc->nbFields; i++)
        {
            if (!strcmp(fc->fields[i].name, "nEmbed"))
                nEmbed = *(int *)fc->fields[i].data;
            if (!strcmp(fc->fields[i].name, "isFp16"))
                isFp16 = *(int *)fc->fields[i].data;
        }
        OneHotPlugin *obj = new OneHotPlugin(nEmbed, isFp16);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
    {
        OneHotPlugin *obj = new OneHotPlugin {serialData, serialLength};
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    const char *getPluginName() const noexcept override
    {
        return "OneHotPlugin";
    }
    const char *getPluginVersion() const noexcept override
    {
        return "1";
    }
    void setPluginNamespace(const char *szNamespace) noexcept override
    {
        mNamespace = szNamespace;
    }
    const PluginFieldCollection *getFieldNames() noexcept override
    {
        return &mFC;
    }
    const char *getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }
};

} // namespace plugin
} // namespace nvinfer1
