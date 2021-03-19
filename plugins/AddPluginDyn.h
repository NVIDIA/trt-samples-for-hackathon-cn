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
#include <iostream>
#include <cstring>
#include <assert.h>

using namespace std;

class AddPluginDyn: public nvinfer1::IPluginV2DynamicExt {
public:
    AddPluginDyn(nvinfer1::Weights valueToAdd) {
        m.valueToAdd = *(float *)valueToAdd.values;
    }
    
    AddPluginDyn(const void *buffer, size_t length) {
        memcpy(&m, buffer, sizeof(m));
    }
    virtual size_t getSerializationSize() const override {
        return sizeof(m);
    }
    virtual void serialize(void *buffer) const override {
        memcpy(buffer, &m, sizeof(m));
    }
    
    nvinfer1::IPluginV2DynamicExt* clone() const override {
        return new AddPluginDyn(&m, sizeof(m));
    }

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override {
    	switch(pos) {
    	case 0:
    		printf("inOut[0].type = %d, format[0]=%d\n", (int)inOut[0].type, (int)inOut[0].format);
    		return 
    			((inOut[0].type == nvinfer1::DataType::kFLOAT || inOut[0].type == nvinfer1::DataType::kHALF) && inOut[0].format == nvinfer1::TensorFormat::kLINEAR)
    			|| (inOut[0].type == nvinfer1::DataType::kINT8 && inOut[0].format == nvinfer1::TensorFormat::kCHW4);
    	case 1:
    		printf("inOut[1].type = %d, format[1]=%d\n", (int)inOut[1].type, (int)inOut[1].format);
    		return inOut[0].format == inOut[1].format && inOut[0].type == inOut[1].type;
    	}
    	return false;
    }

    int getNbOutputs() const override {
        return 1;
    }
    nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs* pInputDim, int nInputDim, nvinfer1::IExprBuilder &exprBuilder) override {
        nvinfer1::DimsExprs ret(pInputDim[0]);
        ret.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *ret.d[0], *exprBuilder.constant(2));
        return ret;
    }
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override {
    	return inputTypes[0] == nvinfer1::DataType::kFLOAT ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kINT8;
    }

    virtual void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInput, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutput) override {
    	m.dataType = in[0].desc.type;
    	m.scale = in[0].desc.scale;
    	printf("configurePlugin type=%d\n", (int)out[0].desc.type);
    }

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc *outputs, int32_t nbOutputs) const override {return 0;}
    int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;

    int initialize() override {return 0;}
    void terminate() override {}
    void destroy() override { delete this; }
    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    const char* getPluginType() const override {return "AddPluginDyn";}
    const char* getPluginVersion() const override {return "0";}
    void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, nvinfer1::IGpuAllocator* /*allocator*/) {}
    void detachFromContext() {}

private:
    using nvinfer1::IPluginV2Ext::configurePlugin;
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
    using nvinfer1::IPluginV2::enqueue;

    struct {
        nvinfer1::DataType dataType;
        float valueToAdd;
        float scale;
    } m;
};

class AddPluginDynCreator : public nvinfer1::IPluginCreator {
public:
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
        return new AddPluginDyn(serialData, serialLength);
    }
    
    const char* getPluginName() const override {return "AddPluginDyn";}
    const char* getPluginVersion() const override {return "0";}

    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    
    const nvinfer1::PluginFieldCollection* getFieldNames() override {
        std::cout << __FUNCTION__ << std::endl;
        return nullptr;
    }
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override {
        std::cout << __FUNCTION__ << std::endl;
        float valueToAdd = 0;
        for (int i = 0; i < fc->nbFields; i++) {
            if (!strcmp(fc->fields[i].name, "valueToAdd")) {
                valueToAdd = *(float *)fc->fields[i].data;
            }
        }
        return new AddPluginDyn({nvinfer1::DataType::kFLOAT, &valueToAdd, 1});
    }
};
