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

#include <vector>
#include <NvInfer.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>

#define DEBUG               0
#define CEIL_DIVIDE(X,Y)    ( ((X) + (Y) - 1) / (Y) )
#define ALIGN32(X)          ( CEIL_DIVIDE((X),32) * 32)

namespace nvinfer1
{
namespace plugin
{

class CumSumPlugin: public IPluginV2DynamicExt
{
private:
    struct
    {
        int nDim;
        int axis;
        int datatype;
        int n;
        int c;
        int h;
        int w;
        int kernelKind;
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
    CumSumPlugin(int axis) 
    {
        m.axis = axis;
    }
    
    CumSumPlugin(const void *buffer, size_t length) 
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
  
    IPluginV2DynamicExt* clone() const override
    {
        return new CumSumPlugin(&m, sizeof(m));
    }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override
    {
        return inputs[0];
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override
    {
        switch(pos)
        {
        case 0:
            return inOut[pos].format == TensorFormat::kLINEAR && ( inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF || inOut[pos].type == DataType::kINT32 );
        case 1:
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == inOut[0].type;
        default:
#if DEBUG
            printf("[CumSumPlugin::supportsFormatCombination()]Error pos!%d\n", pos);
#endif
            return false;
        }
    }
    
    DataType getOutputDataType(int outputIndex, const DataType* inputTypes, int nbInputs) const override
    {
        return inputTypes[0];
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) override
    {
        m.datatype  = int(in[0].desc.type);
        assert(m.datatype == 0 || m.datatype == 1 || m.datatype == 3);
        m.nDim          = in[0].desc.dims.nbDims;
        assert(m.axis >=0 && m.axis < m.nDim);
        m.n             = 1;
        m.c             = 1;
        m.h             = 1;
        m.w             = 1;

        switch(m.nDim)
        {
        case 4:
            m.n = in[0].desc.dims.d[m.nDim - 4];
        case 3:
            m.c = in[0].desc.dims.d[m.nDim - 3];
        case 2:
            m.h = in[0].desc.dims.d[m.nDim - 2];
        case 1:
            m.w = in[0].desc.dims.d[m.nDim - 1];
            break;
        default:
#if DEBUG
            printf("[CumSumPlugin::configurePlugin()]Error input dimension! %d\n", m.nDim);
#endif
            ;
        }

        m.kernelKind    = int(m.w > 32);
    }

    size_t getWorkspaceSize(const PluginTensorDesc* input, int nbInput, const PluginTensorDesc* output, int nbOutput) const override
    {
        return 0;
    }

    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    const char* getPluginType() const override {return "CumSumPlugin";}
    const char* getPluginVersion() const override {return "1";}
    int initialize() override {return 0;}
    void terminate() override {return;}
    void destroy() override {delete this;}
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;
};

class CumSumPluginCreator : public IPluginCreator
{
public:
    CumSumPluginCreator()
    {
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
    {
        int axis = -1;
        for (int i = 0; i < fc->nbFields; i++) 
        {
            if (!strcmp(fc->fields[i].name, "axis"))
                axis = *(int*)fc->fields[i].data;
        }
        return new CumSumPlugin(axis);
    }

    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    const char* getPluginName() const override {return "CumSumPlugin";}
    const char* getPluginVersion() const override {return "1";}
    const PluginFieldCollection* getFieldNames() override {
        std::cout << __FUNCTION__ << std::endl;
        return &mFC;
    }
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {return new CumSumPlugin(serialData, serialLength);}
private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin

} // namespace nvinfer1
