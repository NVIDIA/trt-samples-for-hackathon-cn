/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

#define N_EMBED         128 // should be multiple of 32
#define N_GRID_DIMX     102
#define MASK_VALUE_FP32 9.0e9f
#define MASK_VALUE_FP16 6.0e4f

using namespace nvinfer1;

class LodPreGruPlugin : public IPluginV2DynamicExt
{
private:
    struct
    {
        int datatype;
        int nGroup;
        int nWidth0;
        int nWidth2;
        int nWidth4;
        int nWidth6;
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
    LodPreGruPlugin(int datatype)
    {
        m.datatype = datatype;
    }

    LodPreGruPlugin(const void *buffer, size_t length)
    {
        memcpy(&m, buffer, sizeof(m));
    }

    int getNbOutputs() const override
    {
        return 18;
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
        return new LodPreGruPlugin(&m, sizeof(m));
    }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder) override
    {
        DimsExprs out;
        out.nbDims = 3;
        out.d[0]   = exprBuilder.operation(DimensionOperation::kSUB, *(inputs[7].d[0]), *exprBuilder.constant(1));
        switch (outputIndex)
        {
        case 0:
        case 2:
        case 5:
            out.d[1] = inputs[11].d[0];
            out.d[2] = exprBuilder.constant(128);
            return out;
        case 3:
        case 6:
            out.d[1] = inputs[12].d[0];
            out.d[2] = exprBuilder.constant(128);
            return out;
        case 4:
            out.d[1] = inputs[13].d[0];
            out.d[2] = exprBuilder.constant(128);
            return out;
        case 1:
            out.d[1] = inputs[14].d[0];
            out.d[2] = exprBuilder.constant(128);
            return out;
        case 7:
        case 11:
            out.d[1] = inputs[11].d[0];
            out.d[2] = exprBuilder.constant(1);
            return out;
        case 8:
        case 12:
            out.d[1] = inputs[12].d[0];
            out.d[2] = exprBuilder.constant(1);
            return out;
        case 9:
        case 13:
            out.d[1] = inputs[13].d[0];
            out.d[2] = exprBuilder.constant(1);
            return out;
        case 10:
            out.d[1] = inputs[14].d[0];
            out.d[2] = exprBuilder.constant(1);
            return out;
        case 14:
        case 15:
        case 16:
        case 17:
            out.nbDims = 1;
            return out;
        default:
            printf("[LodPreGruPlugin::getOutputDimensions()]Error outputIndex %d\n", outputIndex);
            return out;
        }
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) override
    {
        switch (pos)
        {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kINT32;
        case 15:
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == ((m.datatype == 0) ? DataType::kFLOAT : DataType::kHALF);
        case 16:
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == inOut[15].type;
        case 30:
        case 31:
        case 32:
        case 33:
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kINT32;
        default:
            printf("[LodPreGruPlugin::supportsFormatCombination()]Error pos!%d\n", pos);
            return false;
        }
    }

    DataType getOutputDataType(int outputIndex, const DataType *inputTypes, int nbInputs) const override
    {
        return (outputIndex < 14) ? ((m.datatype == 0) ? DataType::kFLOAT : DataType::kHALF) : DataType::kINT32;
    }

    void configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs) override
    {
        m.nGroup  = in[7].desc.dims.d[0] - 1;
        m.nWidth0 = in[11].desc.dims.d[0];
        m.nWidth2 = in[12].desc.dims.d[0];
        m.nWidth4 = in[13].desc.dims.d[0];
        m.nWidth6 = in[14].desc.dims.d[0];
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
        return "LodPreGruPlugin";
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

class LodPreGruPluginCreator : public IPluginCreator
{
public:
    IPluginV2DynamicExt *createPlugin(const char *name, const PluginFieldCollection *fc) override
    {
        int datatype = 0;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            if (!strcmp(fc->fields[i].name, "datatype"))
                datatype = *(int *)fc->fields[i].data;
        }
        return new LodPreGruPlugin(datatype);
    }

    const char *getPluginNamespace() const override
    {
        return "";
    }
    const char *getPluginName() const override
    {
        return "LodPreGruPlugin";
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
    IPluginV2DynamicExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override
    {
        return new LodPreGruPlugin(serialData, serialLength);
    }
};
