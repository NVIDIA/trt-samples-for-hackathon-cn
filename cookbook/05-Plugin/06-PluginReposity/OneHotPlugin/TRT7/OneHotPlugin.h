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

#include <NvInfer.h>
#include <cuda_fp16.h>

#define CEIL(x,y) (((x)+(y)-1)/(y))

using namespace nvinfer1;

class OneHotPlugin: public IPluginV2Ext
{
private:
    struct
    {
        int nRow;
        int nEmbed;
        int isFp16;
    } m;

public:
    OneHotPlugin(int nEmbed, int isFp16)
    {
        m.nEmbed = nEmbed;
        m.isFp16 = isFp16;
    }
    
    int getNbOutputs() const override  
    {
        return 1;
    }

    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const
    {
        return m.isFp16? DataType::kHALF : DataType::kFLOAT;
    }
    
    Dims getOutputDimensions(int index, const Dims* pInputDim, int nInputDim) override
    {
        Dims dd = pInputDim[0];
        dd.nbDims += 1;
        dd.d[dd.nbDims -1] = m.nEmbed;
        return dd;
    }
    
    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        return (type == DataType::kINT32 || type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW;
    }

    void configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, const DataType *inputTypes,
                         const DataType *outputTypes, const bool *inputIsBroadcast, const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
    {
        int nRow = 1;
        for(int i = 0; i < inputDims[0].nbDims; i++)
        nRow *= inputDims[0].d[i];
        m.nRow = nRow;        
    }

    int enqueue(int nBatch, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream) override;
    size_t getWorkspaceSize(int nBatch) const override {return 0;}
    OneHotPlugin(const void *buffer, size_t length) {memcpy(&m, buffer, sizeof(m));}    
    IPluginV2Ext* clone() const override {return new OneHotPlugin(&m, sizeof(m));}
    int initialize() override {return 0;}
    void terminate() override {}    
    void destroy() override {delete this;}
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const {return false;}
    bool canBroadcastInputAcrossBatch (int inputIndex) const {return false;}
    void attachToContext (cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) {}
    void detachFromContext() {}
    const char* getPluginType() const override { return "OneHotPlugin";}
    const char* getPluginVersion() const override {return "0";}
    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    virtual size_t getSerializationSize() const override {return sizeof(m);}    
    virtual void serialize(void *buffer) const override {memcpy(buffer, &m, sizeof(m));}
};

class OneHotPluginCreator : public IPluginCreator
{
public:        
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
    {
        int nEmbed = 1, isFp16 = 0;
        for (int i = 0; i < fc->nbFields; i++) 
        {
            if (!strcmp(fc->fields[i].name, "nEmbed"))
                nEmbed = *(int*)fc->fields[i].data;
            if (!strcmp(fc->fields[i].name, "isFp16"))
                isFp16 = *(int*)fc->fields[i].data;
        }
        return new OneHotPlugin(nEmbed, isFp16);
    }

    const char* getPluginName() const override {return "OneHotPlugin";}
    const char* getPluginVersion() const override {return "0";}
    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    const PluginFieldCollection* getFieldNames() override {return nullptr;}
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {return new OneHotPlugin(serialData, serialLength);}
};
