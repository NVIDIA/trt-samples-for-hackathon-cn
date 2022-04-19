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
#include <assert.h>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class GatherND: public IPluginV2DynamicExt {
public:
    GatherND(int batch_dims) {
        m.batch_dims = batch_dims;
    }
    
    GatherND(const void *buffer, size_t length) {
        memcpy(&m, buffer, sizeof(m));
    }

    virtual size_t getSerializationSize() const noexcept override {
        return sizeof(m);
    }
    virtual void serialize(void *buffer) const noexcept override {
        memcpy(buffer, &m, sizeof(m));
    }

    nvinfer1::IPluginV2DynamicExt * clone() const noexcept override {
        return new GatherND(&m, sizeof(m));
    }

    int getNbOutputs() const noexcept override {
        return 1;
    }

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        assert(inputs[1].d[inputs[1].nbDims-1]->isConstant()==true);

        m.index_rank = inputs[1].d[inputs[1].nbDims-1]->getConstantValue();

        m.updatesDim.nbDims = computeUpdatesRank(inputs[0].nbDims, inputs[1].nbDims, m.index_rank, m.batch_dims);
        computeUpdatesShape(&inputs[0], inputs[0].nbDims, &inputs[1], inputs[1].nbDims, m.batch_dims, m.index_rank, m.updatesDim.nbDims, &m.updatesDim);

        return m.updatesDim;
    }

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override
    {
	    return inputs[0].dims.nbDims * sizeof(int32_t);
    }

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int initialize() noexcept override {return 0;}
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(const char* szNamespace) noexcept override {mNamespace = szNamespace;}
    const char* getPluginNamespace() const noexcept override {return mNamespace.c_str();}
    const char* getPluginType() const noexcept override {return "GatherND";}
    const char* getPluginVersion() const noexcept override {return "1";}
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override
    {
        assert(in  && nbInputs  == 2); // the plugin has two input tensors
        assert(out && nbOutputs == 1);
        assert(in[0].desc.type == out[0].desc.type);

        assert( in[0].desc.format == TensorFormat::kLINEAR); //data
        assert( in[1].desc.format == TensorFormat::kLINEAR); //indices
        assert(out[0].desc.format == TensorFormat::kLINEAR);
    }

    //! The combination of kLINEAR + kFLOAT is supported.
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override
    {
        assert(nbInputs == 2 && nbOutputs == 1 && pos < nbInputs + nbOutputs);

        bool condition = inOut[pos].format == TensorFormat::kLINEAR;
        if (pos == 1) // the datatype of the first input is kINT32
            condition &= inOut[pos].type == DataType::kINT32;
        else
            condition &= inOut[pos].type == DataType::kFLOAT;

        return condition;
    }

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        assert(inputTypes && nbInputs == 2);
        return inputTypes[0];
    }

private:

    int multiplyArray(const int *arr, int len) {
        int i,temp=1;
        for(i=0;i<len;i++) {
            temp=temp*arr[i];
        }
        return temp;
    }

    int computeUpdatesRank(int params_rank, int indices_rank, int indices_idx_dim, int batch_dims){
        return params_rank + indices_rank - indices_idx_dim - 1 - batch_dims; 
    }

    void computeUpdatesShape(const nvinfer1::DimsExprs *params_shape, int params_rank, const nvinfer1::DimsExprs *indices_shape, int indices_rank, int batch_dims, int index_rank, int updates_rank, nvinfer1::DimsExprs *updates_shape){
        // to-do (chandler):

        // the batch dims
        for (int i=0; i<batch_dims; ++i){
            updates_shape->d[i] = indices_shape->d[i];
        }
        // the target unit dims
        for (int i=0; i<(indices_rank-batch_dims-1); ++i){
            updates_shape->d[batch_dims+i] = indices_shape->d[batch_dims+i];
        }
        // the index dims
        for (int i=0; i<(params_rank-batch_dims-index_rank); ++i){
            updates_shape->d[indices_rank-1+i] = params_shape->d[batch_dims+index_rank+i];
        }
    }

    struct {
        nvinfer1::DimsExprs updatesDim;
        int batch_dims;
        int index_rank;
    } m;

    const char* mPluginNamespace;
    std::string mNamespace;

    using nvinfer1::IPluginV2Ext::configurePlugin;
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
    using nvinfer1::IPluginV2::enqueue;
};

class GatherNDCreator : public nvinfer1::IPluginCreator {
public:
    GatherNDCreator()
    {
        // TODO: batch_dims is optional in onnx graph
        mPluginAttributes.emplace_back(PluginField("batch_dims", nullptr, PluginFieldType::kINT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override {
        return new GatherND(serialData, serialLength);
    }

    const char* getPluginName() const noexcept override {return "GatherND";}
    const char* getPluginVersion() const noexcept override {return "1";}

    void setPluginNamespace(const char* szNamespace) noexcept override {mNamespace = szNamespace;}
    const char* getPluginNamespace() const noexcept override {return mNamespace.c_str();}

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        std::cout << __FUNCTION__ << std::endl;
        return &mFC;
    }
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override {
        std::cout << __FUNCTION__ << std::endl;
        int batch_dims = 0;
        for (int i = 0; i < fc->nbFields; i++) {
            if (!strcmp(fc->fields[i].name, "batch_dims")) {
                batch_dims = *(int *)fc->fields[i].data;
            }
        }
        GatherND* obj = new GatherND{batch_dims};
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