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

#include "NvInfer.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <assert.h>

namespace nvinfer1
{
namespace plugin
{

class OnehotPlugin: public IPluginV2IOExt {
public:
    OnehotPlugin(int depth) {
        m.depth = depth;
    }
    
    OnehotPlugin(const void *buffer, size_t length) {
        memcpy(&m, buffer, sizeof(m));
    }

    virtual size_t getSerializationSize() const override {
        return sizeof(m);
    }
    virtual void serialize(void *buffer) const override {
        memcpy(buffer, &m, sizeof(m));
    }
    
    nvinfer1::IPluginV2IOExt* clone() const override {
        return new OnehotPlugin(&m, sizeof(m));
    }

    //! The combination of kLINEAR + kFLOAT is supported.
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override
    {
        assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);

        bool condition = inOut[pos].format == TensorFormat::kLINEAR;
        if (pos == 0)
            condition &= inOut[pos].type == nvinfer1::DataType::kINT32;
        else
            condition &= inOut[pos].type == nvinfer1::DataType::kFLOAT;

        return condition;
    }

    int getNbOutputs() const override {
        return 1;
    }
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* pInputDim, int nInputDim) override {
        nvinfer1::Dims dd = pInputDim[0];
        dd.nbDims += 1;
        dd.d[dd.nbDims -1] = m.depth;

        return dd;
    }

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override
    {
        return nvinfer1::DataType::kFLOAT;
    }

    virtual void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override
    {
        assert(in  && nbInput  == 1); 
        assert(out && nbOutput == 1);
        assert( in[0].type == nvinfer1::DataType::kINT32); // the input  tensor is INT32
        assert(out[0].type == nvinfer1::DataType::kFLOAT); // the output tensor is FLOAT
        // Warning: assume the axis is -1
        int nRow = 1;
        for(int i = 0; i < in[0].dims.nbDims; i++)
            nRow *= in[0].dims.d[i];
        m.nRow = nRow;
    }

    size_t getWorkspaceSize(int nBatch) const override {return 0;}
    int enqueue(int nBatch, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream) override;

    int initialize() override {return 0;}
    void terminate() override {}
    void destroy() override { delete this; }
    void setPluginNamespace(const char* szNamespace) override {mNamespace = szNamespace;}
    const char* getPluginNamespace() const override {return mNamespace.c_str();}
    const char* getPluginType() const override {return "OnehotPlugin";}
    const char* getPluginVersion() const override {return "1";}
    bool canBroadcastInputAcrossBatch(int inputIndex) const override {return false;}
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const {return false;}
    void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, nvinfer1::IGpuAllocator* /*allocator*/) {}
    void detachFromContext() {}

private:
    struct {
        int depth;
	    int nRow;
    } m;
    const char* mPluginNamespace;
    std::string mNamespace;

    using nvinfer1::IPluginV2Ext::configurePlugin;
};

class OnehotPluginCreator: public nvinfer1::IPluginCreator {
public:
    OnehotPluginCreator(){
        mPluginAttributes.emplace_back(PluginField("depth", nullptr, PluginFieldType::kINT32, 1));
	    mFC.nbFields = mPluginAttributes.size();
	    mFC.fields = mPluginAttributes.data();
    }
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
        OnehotPlugin* obj = new OnehotPlugin{serialData, serialLength};
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    
    const char* getPluginName() const override {return "OnehotPlugin";}
    const char* getPluginVersion() const override {return "1";}

    void setPluginNamespace(const char* szNamespace) override {mNamespace = szNamespace;}
    const char* getPluginNamespace() const override {return mNamespace.c_str();}
    
    const nvinfer1::PluginFieldCollection* getFieldNames() override {
        std::cout << __FUNCTION__ << std::endl;
        return &mFC;
    }
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override {
        std::cout << __FUNCTION__ << std::endl;
        int depth = 0;
        for (int i = 0; i < fc->nbFields; i++) {
            if (!strcmp(fc->fields[i].name, "depth")) {
                depth = *(int *)fc->fields[i].data;
            }
        }
        OnehotPlugin* obj = new OnehotPlugin{depth};
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin

} // namespace nvinfer1