#include <vector>
#include <string>
#include <NvInfer.h>
#include <cub/cub.cuh>

#define CEIL_DIVIDE(X,Y)    (((X)+(Y)-1)/(Y))
#define CEIL_TO(X,Y)        (((X)+(Y)-1)/(Y)*(Y))

namespace nvinfer1
{

template <typename T>
class LayerNormPlugin: public IPluginV2DynamicExt
{
private:    
    std::string name_;
    std::string namespace_;

public:
    LayerNormPlugin(const std::string& name) : name_(name)
    {
        DEBUG_FUNC();
    }

    LayerNormPlugin(const std::string& name, const void* data, size_t length) : name_(name)
    {
        DEBUG_FUNC();
    }
    
    LayerNormPlugin() = delete;

    ~LayerNormPlugin()
    {
        DEBUG_FUNC();
    }

    size_t getSerializationSize() const noexcept override
    {
        DEBUG_FUNC();
        return 0;
    }
    
    void serialize(void *buffer) const noexcept override
    {
        DEBUG_FUNC();
    }
  
    IPluginV2DynamicExt* clone() const noexcept override
    {
        DEBUG_FUNC();
        return new LayerNormPlugin<T>(name_);
    }

    int getNbOutputs() const noexcept override
    {
        DEBUG_FUNC();
        return 1;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        DEBUG_FUNC();
        return inputs[0];
    }

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        DEBUG_FUNC();
        if(inOut[pos].format != TensorFormat::kLINEAR)
        {
            return false;
        }

        bool res = false;
        switch(pos)
        {
        case 0:
            res = inOut[pos].type == DataType::kFLOAT; break;
        case 1:
            res = inOut[pos].type == DataType::kFLOAT; break;
        case 2:
            res = inOut[pos].type == DataType::kFLOAT; break;
        case 1:
            res = inOut[pos].type == DataType::kFLOAT; break;
        default:// should NOT be here
            break;
        }
        return res;
    }
    
    DataType getOutputDataType(int outputIndex, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        DEBUG_FUNC();
        return inputTypes[0];
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override
    {
        DEBUG_FUNC();
    }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,int32_t nbOutputs) const noexcept override
    {
        DEBUG_FUNC();
        return 0;
    }

    void setPluginNamespace(const char* szNamespace) noexcept override
    {
        DEBUG_FUNC();
        namespace_ = szNamespace;
    }
    const char* getPluginNamespace() const noexcept override
    {
        DEBUG_FUNC();
        return namespace_.c_str();
    }
    const char* getPluginType() const noexcept override
    {
        DEBUG_FUNC();
        return PLUGIN_NAME;
    }
    const char* getPluginVersion() const noexcept override
    {
        DEBUG_FUNC();
        return PLUGIN_VERSION;
    }
    int initialize() noexcept override
    {
        DEBUG_FUNC();
        return 0;
    }
    void terminate() noexcept override
    {
        DEBUG_FUNC();
        return;
    }

    void destroy() noexcept override
    {
        DEBUG_FUNC();
    }
    
    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
}; // class LayerNormPlugin

class LayerNormPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection fc_;
    static std::vector<PluginField> attr_;
    std::string namespace_;

public:
    LayerNormPluginCreator()
    {
        attr_.emplace_back(PluginField("prevScale", nullptr, PluginFieldType::kFLOAT32,1));
        attr_.emplace_back(PluginField("postScale", nullptr, PluginFieldType::kFLOAT32,1));
        fc_.nbFields = attr_.size();
        fc_.fields = attr_.data();
    }

    ~LayerNormPluginCreator() {}

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
    {
        DEBUG_FUNC();
        return new LayerNormPlugin<float>(name, prevScale, postScale);
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        DEBUG_FUNC();
        return new LayerNormPlugin<float>(name, serialData, serialLength);
    }

    void setPluginNamespace(const char* szNamespace) noexcept override
    {
        namespace_ = szNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return namespace_.c_str();
    }

    const char* getPluginName() const noexcept override
    {
        return PLUGIN_NAME;
    }

    const char* getPluginVersion() const noexcept override
    {
        return PLUGIN_VERSION;
    }

    const PluginFieldCollection* getFieldNames() noexcept override
    {
        return &fc_;
    }
}; // class LayerNormPluginCreator

} // namespace nvinfer1

