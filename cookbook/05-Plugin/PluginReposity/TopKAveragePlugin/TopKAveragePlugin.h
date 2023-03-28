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
#include <cuda_fp16.h>

#define ALIGNSIZE 1024

#define CEIL(X, Y) (((X) + (Y)-1) / (Y))
#define ALIGNED(X) (CEIL((X), (ALIGNSIZE)) * (ALIGNSIZE))

using namespace nvinfer1;

class TopKAveragePlugin : public IPluginV2DynamicExt
{
private:
    struct
    {
        int datatype;
        int nTopK;
        int maxTopK;
        int nGroup;
        int nChannel;
        int nHeight;
        int nWidth;
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
    TopKAveragePlugin(int nTopK, int maxTopK)
    {
        m.nTopK   = nTopK;
        m.maxTopK = maxTopK;
    }

    TopKAveragePlugin(const void *buffer, size_t length)
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
        return new TopKAveragePlugin(&m, sizeof(m));
    }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder) override
    {
        DimsExprs temp;
        temp.nbDims = 1;
        temp.d[0]   = exprBuilder.constant(m.nTopK);

        DimsExprs output;
        output.nbDims = 3;
        output.d[0]   = inputs[0].d[0];
        output.d[1]   = inputs[0].d[2];
        output.d[2]   = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[1], *temp.d[0]);
        return output;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) override
    {
        switch (pos)
        {
        case 0: return inOut[0].format == TensorFormat::kLINEAR && (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF);
        case 1: return inOut[1].format == TensorFormat::kLINEAR && inOut[1].type == DataType::kINT32;
        case 2: return inOut[2].format == TensorFormat::kLINEAR && inOut[2].type == DataType::kINT32;
        case 3: return inOut[3].format == TensorFormat::kLINEAR && inOut[3].type == DataType::kINT32;
        case 4: return inOut[4].format == TensorFormat::kLINEAR && inOut[4].type == inOut[0].type;
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
        m.nChannel = in[0].desc.dims.d[1];
        m.nHeight  = in[0].desc.dims.d[2];
        m.nWidth   = in[0].desc.dims.d[3];
    }

    size_t getWorkspaceSize(const PluginTensorDesc *input, int nbInput, const PluginTensorDesc *output, int nbOutput) const override
    {
        return ALIGNED(sizeof(int32_t) * input[0].dims.d[0] * input[0].dims.d[1] * input[0].dims.d[2] * input[0].dims.d[3] * m.maxTopK);
    }

    int enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;

    int initialize() override
    {
        return 0;
    }
    void        terminate() override {}
    const char *getPluginType() const override
    {
        return "TopKAveragePlugin";
    }
    const char *getPluginVersion() const override
    {
        return "0";
    }
    void destroy() override
    {
        delete this;
    }
    void        setPluginNamespace(const char *szNamespace) override {}
    const char *getPluginNamespace() const override
    {
        return "";
    }
};

class TopKAveragePluginCreator : public IPluginCreator
{
public:
    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override
    {
        return new TopKAveragePlugin(serialData, serialLength);
    }

    const char *getPluginName() const override
    {
        return "TopKAveragePlugin";
    }

    const char *getPluginVersion() const override
    {
        return "0";
    }

    void setPluginNamespace(const char *szNamespace) override {}

    const char *getPluginNamespace() const override
    {
        return "";
    }

    const PluginFieldCollection *getFieldNames() override
    {
        return nullptr;
    }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override
    {
        int nTopK = 1, maxTopK = 1;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            if (!strcmp(fc->fields[i].name, "nTopK"))
                nTopK = *(int *)fc->fields[i].data;
            if (!strcmp(fc->fields[i].name, "maxTopK"))
                maxTopK = *(int *)fc->fields[i].data;
        }
        return new TopKAveragePlugin(nTopK, maxTopK);
    }
};
