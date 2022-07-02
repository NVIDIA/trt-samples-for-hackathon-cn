#include <NvInfer.h>
#include <algorithm>
#include <cstring>

using namespace nvinfer1;

class LayerNormPlugin: public IPluginV2DynamicExt
{
public:
    LayerNormPlugin() {}
    
    LayerNormPlugin(const void *buffer, size_t length) 
    {
    }
    
    int getNbOutputs() const override
    {
        return 1;
    }
  
    IPluginV2DynamicExt* clone() const override
    {
        return new LayerNormPlugin();
    }

    virtual size_t getSerializationSize() const override
    {
        return 0;
    }
    
    virtual void serialize(void *buffer) const override
    {
    }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override
    {
        DimsExprs output{1};
        output.d[0] = exprBuilder.constant(inputs->nbDims);
        return output;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override
    {
        bool ret;
        switch(pos)
        {
        case 0:
    		ret = inOut[0].format == nvinfer1::TensorFormat::kLINEAR &&
                  (inOut[0].type == nvinfer1::DataType::kFLOAT || inOut[0].type == nvinfer1::DataType::kHALF || inOut[0].type == nvinfer1::DataType::kINT8);
            break;            
    	case 1:
    		ret = inOut[1].format == nvinfer1::TensorFormat::kLINEAR && inOut[1].type == nvinfer1::DataType::kINT32 &&
                  (inOut[0].type == nvinfer1::DataType::kFLOAT || inOut[0].type == nvinfer1::DataType::kHALF || inOut[0].type == nvinfer1::DataType::kINT8);
            break;
        }
        return ret;
    }
    
    DataType getOutputDataType(int outputIndex, const DataType* inputTypes, int nbInputs) const override
    {
    	return DataType::kINT32;
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) override {}

    int initialize() override {return 0;}
    void terminate() override {}

    size_t getWorkspaceSize(const PluginTensorDesc* input, int nbInput, const PluginTensorDesc* output, int nbOutput) const override {return 0;}
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;
    const char* getPluginType() const override {return "ShapePlugin";}
    const char* getPluginVersion() const override {return "0";}
    void destroy() override {delete this;}    
    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}    
};

class AddPluginCreator : public nvinfer1::IPluginCreator
{
public:
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
    {
        return new LayerNormPlugin(serialData, serialLength);
    }
    
    const char* getPluginName() const override
    {
        return "ShapePlugin";
    }
    
    const char* getPluginVersion() const override 
    {
        return "0";
    }

    void setPluginNamespace(const char* szNamespace) override {}
    
    const char* getPluginNamespace() const override
    {
        return "";
    }
    
    const nvinfer1::PluginFieldCollection* getFieldNames() override
    {
        return nullptr;
    }
    
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override
    {
        return new LayerNormPlugin();
    }
};
