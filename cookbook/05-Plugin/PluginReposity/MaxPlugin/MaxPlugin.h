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

#define SMALL_NUMBER -60000

using namespace nvinfer1;

class MaxPlugin : public IPluginV2DynamicExt
{
private:
    struct
    {
        int datatype;
        int nGroup;
        int nWidth;
        int nEmbed;
    } m;

protected:
    // Prevent warning
    using IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using IPluginV2DynamicExt::configurePlugin;
    using IPluginV2DynamicExt::enqueue;
    using IPluginV2DynamicExt::getOutputDimensions;
    using IPluginV2DynamicExt::getWorkspaceSize;
    using IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using IPluginV2DynamicExt::supportsFormat;

public:
    MaxPlugin() {}

    MaxPlugin(const void *buffer, size_t length)
    {
        memcpy(&m, buffer, sizeof(m));
    }

    int getNbOutputs() const override
    {
        return 1;
    }

    virtual size_t getSerializationSize() const override
    {
        return sizeof(m);
    }

    virtual void serialize(void *buffer) const override
    {
        memcpy(buffer, &m, sizeof(m));
    }

    IPluginV2DynamicExt *clone() const override
    {
        return new MaxPlugin(&m, sizeof(m));
    }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder) override
    {
        DimsExprs out;
        out.nbDims = 2;
        out.d[0]   = inputs[0].d[0];
        out.d[1]   = inputs[0].d[2];
        return out;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) override
    {
        switch (pos)
        {
        case 0: return inOut[0].format == TensorFormat::kLINEAR && (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF || inOut[0].type == DataType::kINT32);
        case 1: return inOut[1].format == TensorFormat::kLINEAR && inOut[1].type == DataType::kINT32;
        case 2: return inOut[2].format == TensorFormat::kLINEAR && inOut[2].type == inOut[0].type;
        }
        return false;
    }

    DataType getOutputDataType(int outputIndex, const DataType *inputTypes, int nbInputs) const override
    {
        return inputTypes[0];
    }

    void configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs) override
    {
        m.datatype = int(in[0].desc.type);
        m.nGroup   = in[0].desc.dims.d[0];
        m.nWidth   = in[0].desc.dims.d[1];
        m.nEmbed   = in[0].desc.dims.d[2];
    }

    size_t getWorkspaceSize(const PluginTensorDesc *input, int nbInput, const PluginTensorDesc *output, int nbOutput) const override
    {
        return 0;
    }
    const char *getPluginNamespace() const override
    {
        return "";
    }
    const char *getPluginType() const override
    {
        return "MaxPlugin";
    }
    const char *getPluginVersion() const override
    {
        return "0";
    }
    void setPluginNamespace(const char *szNamespace) override {}
    int  initialize() override
    {
        return 0;
    }
    void terminate() override
    {
        return;
    }
    void destroy() override
    {
        delete this;
    }
    int enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;
};

class MaxPluginCreator : public IPluginCreator
{
public:
    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override
    {
        return new MaxPlugin();
    }

    const char *getPluginNamespace() const override
    {
        return "";
    }
    const char *getPluginName() const override
    {
        return "MaxPlugin";
    }
    const char *getPluginVersion() const override
    {
        return "0";
    }
    void                         setPluginNamespace(const char *szNamespace) override {}
    const PluginFieldCollection *getFieldNames() override
    {
        return nullptr;
    }
    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override
    {
        return new MaxPlugin(serialData, serialLength);
    }
};
