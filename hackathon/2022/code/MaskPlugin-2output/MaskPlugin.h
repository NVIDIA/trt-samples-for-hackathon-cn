#include <vector>
#include <string>
#include <cassert>
#include <NvInfer.h>
#include <cuda_fp16.h>

#define CEIL_DIVIDE(X,Y)    (((X)+(Y)-1)/(Y))
#define CEIL_TO(X,Y)        (((X)+(Y)-1)/(Y)*(Y))

// +------- Debug wrapper --------------------------------------------------------------------------
#if DEBUG
#define WHERE_AM_I() do {printf("[%s]: this=->%p\n",__func__,this);} while(0);
#else
#define WHERE_AM_I()
#endif // DEBUG

template <typename T>
__device__ T negtiveInfinity();

template <>
__device__ float negtiveInfinity <float>()
{
    return (float)-6.0e6;
}

template <>
__device__ half negtiveInfinity<half>()
{
    return (half)-6.0e6;
}

// +------- Plguin ---------------------------------------------------------------------------------
namespace
{
static const char* PLUGIN_NAME{"Mask"};
static const char* PLUGIN_VERSION{"1"};
} // namespace

namespace nvinfer1
{

// +------- Plugin body ----------------------------------------------------------------------------
class MaskPlugin: public IPluginV2DynamicExt
{
private:    
    std::string name_;
    std::string namespace_;

public:
    MaskPlugin(const std::string& name) : name_(name)
    {
        WHERE_AM_I();
    }

    MaskPlugin(const std::string& name, const void* data, size_t length) : name_(name)
    {
        WHERE_AM_I();
    }
    
    MaskPlugin() = delete;

    ~MaskPlugin()
    {
        WHERE_AM_I();
    }

    size_t getSerializationSize() const noexcept override
    {
        WHERE_AM_I();
        return 0;
    }
    
    void serialize(void *buffer) const noexcept override
    {
        WHERE_AM_I();
    }
  
    IPluginV2DynamicExt* clone() const noexcept override
    {
        WHERE_AM_I();
        return new MaskPlugin(name_);
    }

    int getNbOutputs() const noexcept override
    {
        WHERE_AM_I();
        return 2;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        WHERE_AM_I();
        switch(outputIndex)
        {
        case 0:
        case 1:
            DimsExprs ret;
            ret.nbDims = 4;
            ret.d[0] = inputs[0].d[0];
            ret.d[1] = exprBuilder.constant(1);
            ret.d[2] = exprBuilder.constant(1);
            ret.d[3] = exprBuilder.operation(DimensionOperation::kSUB,
                                                *exprBuilder.operation(DimensionOperation::kFLOOR_DIV,
                                                                        *inputs[0].d[1],
                                                                        *exprBuilder.constant(4)
                                                                        ),
                                                *exprBuilder.constant(1)
                                            );
            return ret;
        default: // should NOT be here
            return inputs[0];
        }
    }

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        WHERE_AM_I();
        if(inOut[pos].format != TensorFormat::kLINEAR)
        {
            return false;
        }

        bool res = false;
        switch(pos)
        {
        case 0:
            res = inOut[pos].type == DataType::kFLOAT   && inOut[pos].dims.nbDims == 3; break;
        case 1:
            res = inOut[pos].type == DataType::kINT32   && inOut[pos].dims.nbDims == 1; break;
        case 2:
            res = inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF; break;
        case 3:
            res = inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF; break;
        default:// should NOT be here
            break;
        }
#if DEBUG
        printf("pos=%d,res=%d,[%d,%d,%d,%d]\n", pos,
                                                int(res),
                                                int(inOut[0].type),
                                                int(inOut[1].type),
                                                int(inOut[2].type),
                                                int(inOut[3].type));
#endif
        return res;
    }
    
    DataType getOutputDataType(int outputIndex, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        WHERE_AM_I();
        //return DataType::kFLOAT;
        return DataType::kHALF;
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override
    {
        WHERE_AM_I();
    }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,int32_t nbOutputs) const noexcept override
    {
        WHERE_AM_I();
        return 0;
    }

    void setPluginNamespace(const char* szNamespace) noexcept override
    {
        WHERE_AM_I();
        namespace_ = szNamespace;
    }
    const char* getPluginNamespace() const noexcept override
    {
        WHERE_AM_I();
        return namespace_.c_str();
    }
    const char* getPluginType() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_NAME;
    }
    const char* getPluginVersion() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_VERSION;
    }
    int initialize() noexcept override
    {
        WHERE_AM_I();
        return 0;
    }
    void terminate() noexcept override
    {
        WHERE_AM_I();
        return;
    }

    void destroy() noexcept override
    {
        WHERE_AM_I();
    }
    
    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
}; // class MaskPlugin

class MaskPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection fc_;
    static std::vector<PluginField> attr_;
    std::string namespace_;

public:
    MaskPluginCreator()
    {
        fc_.nbFields = attr_.size();
        fc_.fields = attr_.data();
    }

    ~MaskPluginCreator() {}

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
    {
        WHERE_AM_I();
        return new MaskPlugin(name);
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        return new MaskPlugin(name, serialData, serialLength);
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
}; // class MaskPluginCreator

} // namespace nvinfer1

