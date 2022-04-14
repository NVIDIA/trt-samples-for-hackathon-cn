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
#include <cub/cub.cuh>
#include <curand.h>
#include <cstring>
#include <vector>

#define ALIGNSIZE  1024
#define ALIGNED(x) (((x) + ALIGNSIZE - 1) / ALIGNSIZE * ALIGNSIZE)

namespace nvinfer1
{
namespace plugin
{    

class RandomPlugin : public IPluginV2DynamicExt
{
private:
    struct
    {
        int                nRow;
        int                nCol;
        unsigned long long seed;
    } m;
    std::string mNamespace;

public:
    RandomPlugin(int seed)
    {
        m.seed = (unsigned long long)seed;
    }

    int getNbOutputs() const noexcept override
    {
        return 2;
    }

    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept
    {
        return (index == 0) ? DataType::kINT32 : DataType::kFLOAT;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        DimsExprs out(inputs[0]);
        out.nbDims = 2;
        out.d[1] = exprBuilder.constant(1);
        return out;
    }

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);

        bool condition = true;
        if (pos == 1)
            condition &= inOut[pos].type == nvinfer1::DataType::kINT32;
        else if(pos == 0) 
            condition &= inOut[pos].type == nvinfer1::DataType::kFLOAT;

        return condition;
    }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,int32_t nbOutputs) const noexcept override
    {
        return ALIGNED(m.nRow * sizeof(float)) + ((m.nCol == 192) ? ALIGNED(m.nRow * 192 * sizeof(int)) : 0);
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override
    {   
        m.nRow = in[0].max.d[0];
        m.nCol = in[0].max.d[1];
    }

    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    RandomPlugin(const void *buffer, size_t length)
    {
        memcpy(&m, buffer, sizeof(m));
    }
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override
    {
        return new RandomPlugin(&m, sizeof(m));
    }
    int initialize() noexcept override
    {
        return 0;
    }
    void terminate() noexcept override {}
    void destroy() noexcept override
    {
        delete this;
    }
    
    void        attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept {}
    void        detachFromContext() noexcept {}
    const char *getPluginType() const noexcept override
    {
        return "RandomPlugin";
    }
    const char *getPluginVersion() const noexcept override
    {
        return "1";
    }
    void        setPluginNamespace(const char *szNamespace) noexcept override {mNamespace = szNamespace;}
    const char *getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }
    virtual size_t getSerializationSize() const noexcept override
    {
        return sizeof(m);
    }
    virtual void serialize(void *buffer) const noexcept override
    {
        memcpy(buffer, &m, sizeof(m));
    }
};

class RandomPluginCreator : public IPluginCreator
{
public:

    RandomPluginCreator(){
        mPluginAttributes.emplace_back(PluginField("seed", nullptr, PluginFieldType::kINT32, 1));
	    mFC.nbFields = mPluginAttributes.size();
	    mFC.fields = mPluginAttributes.data();
    }
    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
    {
        int seed = 97;
        for (int i = 0; i < fc->nbFields; i++)
        {
            if (!strcmp(fc->fields[i].name, "seed"))
                seed = *(int *)fc->fields[i].data;
        }
        RandomPlugin* obj = new RandomPlugin(seed);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    const char *getPluginName() const noexcept override
    {
        return "RandomPlugin";
    }
    const char *getPluginVersion() const noexcept override
    {
        return "1";
    }
    void        setPluginNamespace(const char *szNamespace) noexcept override {mNamespace = szNamespace;}
    const char *getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();;
    }
    const PluginFieldCollection *getFieldNames() noexcept override
    {
        return &mFC;
    }
    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
    {
        RandomPlugin* obj = new RandomPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};
}
}
