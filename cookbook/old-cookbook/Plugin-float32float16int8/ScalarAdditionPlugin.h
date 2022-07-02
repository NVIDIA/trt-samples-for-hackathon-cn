// ElementAdditionPlugin.h
#include <NvInfer.h>
//#include <cstdio>
#include <cstring>

#define CEIL(X,Y) ( ((X) + (Y) - 1) / (Y) )

class ScalarAdditionPlugin: public nvinfer1::IPluginV2IOExt
{
private:
    struct
    {
        nvinfer1::DataType  dataType;
        nvinfer1::Dims      inputDim;
        float               addend;
        float               scale;
    } m;

public:
    ScalarAdditionPlugin(nvinfer1::Weights addend)
    {
        m.addend = *(float *)addend.values;
    }
    
    ScalarAdditionPlugin(const void *buffer, size_t length) 
    {
        memcpy(&m, buffer, sizeof(m));
    }
    
    virtual size_t getSerializationSize() const override
    {
        return sizeof(m);
    }
    
    virtual void serialize(void *buffer) const override
    {
        memcpy(buffer, &m, sizeof(m));
    }
    
    nvinfer1::IPluginV2IOExt* clone() const override
    {
        return new ScalarAdditionPlugin(&m, sizeof(m));
    }

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override
    {
		//printf("inout[0](dataType,format)=(%d,%d),inout[1](dataType,format)=(%d,%d)\n", inOut[0].type,inOut[0].format,inOut[1].type,inOut[1].format);
        //printf("type:%d,%d,%d\n",nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF,nvinfer1::DataType::kINT8);
        //printf("format:%d,%d,%d\n",nvinfer1::TensorFormat::kLINEAR,nvinfer1::TensorFormat::kCHW4,nvinfer1::TensorFormat::kCHW32);
        bool ret;
        switch(pos)
        {
        case 0:
    		ret = inOut[0].format == nvinfer1::TensorFormat::kLINEAR &&
                  (inOut[0].type == nvinfer1::DataType::kFLOAT || inOut[0].type == nvinfer1::DataType::kHALF || inOut[0].type == nvinfer1::DataType::kINT8);
            break;            
    	case 1:
    		ret = inOut[1].format == nvinfer1::TensorFormat::kLINEAR && 
                  inOut[0].type == nvinfer1::DataType::kFLOAT && inOut[1].type == nvinfer1::DataType::kFLOAT ||
                  inOut[0].type == nvinfer1::DataType::kHALF && inOut[1].type == nvinfer1::DataType::kHALF ||
                  inOut[0].type == nvinfer1::DataType::kINT8 && inOut[1].type == nvinfer1::DataType::kINT8;
            break;
        }
        //printf("ret[%d]=%d\n", pos, ret);
        return ret;
    }
    
    int getNbOutputs() const override
    {
        return 1;
    }
    
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* pInputDim, int nInputDim) override
    {
        return pInputDim[0];
    }

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override
    {
    	return inputTypes[0];
    }
	
    virtual void configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInput, const nvinfer1::PluginTensorDesc* out, int nbOutput) override
    {
        m.dataType = in[0].type;
        m.inputDim = in[0].dims;
    	m.scale    = in[0].scale;
        //printf("configurePlugin (dataType,scale)=(%d,%f)\n", int(m.dataType),m.scale);
    }

    size_t getWorkspaceSize(int nBatch) const override
    {
        return 0;                                                                                   // 临时工作空间大小（Byte）
    }
    
    int enqueue(int nBatch, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream) override;
    int initialize() override {return 0;}    
    void terminate() override {}    
    void destroy() override {delete this;}    
    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}    
    const char* getPluginType() const override {return "ScalarAdditionPlugin";}    
    const char* getPluginVersion() const override {return "0";}
    bool canBroadcastInputAcrossBatch(int inputIndex) const override {return false;}
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const {return false;}
    void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, nvinfer1::IGpuAllocator* /*allocator*/) {}
    void detachFromContext() {}
};

class ScalarAdditionPluginCreator : public nvinfer1::IPluginCreator
{
public:
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
    {
        return new ScalarAdditionPlugin(serialData, serialLength);
    }
    
    const char* getPluginName() const override
    {
        return "ScalarAdditionPlugin";
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
        float addend = 0.0f;
        for (int i = 0; i < fc->nbFields; i++) 
        {
            if (!strcmp(fc->fields[i].name, "addend"))
                addend = *(float*)fc->fields[i].data;
        }
        return new ScalarAdditionPlugin({nvinfer1::DataType::kFLOAT, &addend, 1});
    }
};
