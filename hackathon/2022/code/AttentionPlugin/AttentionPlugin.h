#include "utils.h"
#include "layerKernels.h"
#include "attentionKernels.h"

#include <vector>
#include <string>
#include <cassert>
#include <NvInfer.h>

const char* parameterFile       = "./para.npz";
const char* parameterFileFP16   = "./para-fp16.npz";
const char* gemm_file           = "./gemm.in";

#ifdef USE_FP_16
#if USE_FP_16 == 1
const bool  useFP16             = true;
#else   // # if
const bool  useFP16             = false;
#endif  // # if
#else   // # ifdef
const bool  useFP16             = false;
#endif  // # ifdef

// +------- Debug wrapper --------------------------------------------------------------------------
#if DEBUG_ENABLE
#define DEBUG_FUNC() do {printf("[%s]: this=->%p\n",__func__,this);} while(0);

// print the first 10 elements in buffer "X_"
#define PRINT(X_, IS_FP) do {                                                   \
                                cudaDeviceSynchronize();                        \
                                printf(#X_"[%p]=\n\t",X_);                      \
                                (print<T>)<<<1,1,0,stream>>>((void*)X_, IS_FP); \
                                cudaDeviceSynchronize();                        \
                            } while(0);

// use macro PRINT to print all device buffer
#define BIG_PRINT_WEIGHT()                      \
    do                                          \
    {                                           \
        printf("\n+--------WEIGHT--------\n");  \
        PRINT(pos_bias_u_dev_,true)             \
        PRINT(pos_bias_v_dev_,true)             \
        PRINT(linear_q_weight_dev_,true)        \
        PRINT(linear_q_bias_dev_,true)          \
        PRINT(linear_k_weight_dev_,false)       \
        PRINT(linear_k_bias_dev_,true)          \
        PRINT(linear_v_weight_dev_,true)        \
        PRINT(linear_v_bias_dev_,true)          \
        PRINT(linear_out_weight_dev_,true)      \
        PRINT(linear_out_bias_dev_,true)        \
        PRINT(linear_pos_weight_dev_,true)      \
    } while(0);

#define BIG_PRINT(i)                            \
    do                                          \
    {                                           \
        printf("\n+--------%2d--------\n",i);   \
        PRINT(x_q,true)                         \
        PRINT(pos_emb,true)                     \
        PRINT(mask,false)                       \
        PRINT(q,true)                           \
        PRINT(k,true)                           \
        PRINT(v,true)                           \
        PRINT(p,true)                           \
        PRINT(k_transpose,true)                 \
        PRINT(v_transpose,true)                 \
        PRINT(p_transpose,true)                 \
        PRINT(q_with_bias_u,true)               \
        PRINT(q_with_bias_v,true)               \
        PRINT(matrix_ac,true)                   \
        PRINT(matrix_bd,true)                   \
        PRINT(score,true)                       \
        PRINT(x,true)                           \
        PRINT(x_transpose,true)                 \
        PRINT(res,true)                         \
    } while(0);
//                            PRINT(q_transpose,true)             \ // wili, delete this output

#else
#define BIG_PRINT_WEIGHT()
#define DEBUG_FUNC()
#define BIG_PRINT(i)
#endif // DEBUG_ENABLE

template <int NUM>
__forceinline__ __device__ __host__ int round(int num)
{
    return ((num - 1) / NUM + 1) * NUM;
}

template <typename T>
__global__ void print(const void *, bool);

template <>
__global__ void print<float>(const void * input0, bool isFP)
{
    if (isFP)
    {
        float* input = (float*)input0;
        for(int i=0;i<10;i++)
            printf("%.3f,", input[i]);
       printf("\n");
    }
    else
    {
        int* input = (int*)input0;
        for(int i=0;i<10;i++)
            printf("%2d,", input[i]);
       printf("\n");
    }
}

template <>
__global__ void print<half>(const void * input0, bool isFP)
{
    if (isFP)
    {
        half* input = (half*)input0;
        for(int i=0;i<10;i++)
            printf("%.3f,", __half2float(input[i]));
       printf("\n");
    }
    else
    {
        int* input = (int*)input0;
        for(int i=0;i<10;i++)
            printf("%2d,", input[i]);
       printf("\n");
    }
}

// +------- Plguin ---------------------------------------------------------------------------------
namespace
{
static const char* PLUGIN_NAME{"Attention"};
static const char* PLUGIN_VERSION{"1"};
} // namespace

// +------- cuBLAS wrapper -------------------------------------------------------------------------
template <typename T>
cudaDataType_t cublas_dtype();

template <>
cudaDataType_t cublas_dtype<float>()
{
    return CUDA_R_32F;
}

template <>
cudaDataType_t cublas_dtype<half>()
{
    return CUDA_R_16F;
}

int cublas_dtype_size(cudaDataType_t dtype)
{
    if (dtype == CUDA_R_32F)
        return 4;
    if (dtype == CUDA_R_16F)
        return 2;
    assert(0); // should NOT be here
    return 0;
}

namespace nvinfer1
{

// +------- TRT wrapper ----------------------------------------------------------------------------
template <typename T>
DataType trt_dtype();

template <>
DataType trt_dtype<float>()
{
    return DataType::kFLOAT;
}

template <>
DataType trt_dtype<half>()
{
    return DataType::kHALF;
}

int trt_dtype_size(DataType dtype)
{
    switch(dtype)
    {
    case DataType::kFLOAT:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT32:
        return 4;
    default:// should NOT be here
        assert(0); 
    }
    return 0;
}

template <typename T>
inline size_t trt_serialize_size(T value)
{
    return sizeof(value);
}

template <typename T>
inline size_t trt_serialize_size(const std::vector<T>& value)
{
    return sizeof(value.size()) + value.size()*sizeof(T);
}

inline size_t trt_serialize_size(const Weights& w)
{
    return sizeof(w.type) + sizeof(w.count) + w.count * trt_dtype_size(w.type);
}

template <typename T>
inline void trt_serialize_value(void** buffer, T value)
{
    T* ptr = reinterpret_cast<T*>(*buffer);
    *ptr = value;
    uintptr_t addr = reinterpret_cast<uintptr_t>(*buffer);
    addr += sizeof(T);
    *buffer = reinterpret_cast<void*>(addr);
}

inline void trt_serialize_value(void** buffer, const void* value, size_t size)
{
    void* ptr = *buffer;
    memcpy(ptr, value, size);
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    addr += size;
    *buffer = reinterpret_cast<void*>(addr);
}

template <typename T>
inline void trt_serialize_value(void** buffer, const std::vector<T>& value)
{
    trt_serialize_value(buffer, value.size());
    size_t size = value.size() * sizeof(T);
    trt_serialize_value(buffer, value.data(), size);
}

inline void trt_serialize_value(void** buffer, const Weights& w)
{
    trt_serialize_value(buffer, w.type);
    trt_serialize_value(buffer, w.count);
    size_t size = w.count * trt_dtype_size(w.type);
    trt_serialize_value(buffer, w.values, size);
}

template <typename T>
inline void trt_deserialize_value(const void* data, size_t length, size_t& offset, T& value)
{
    assert(offset < length);
    uintptr_t addr = reinterpret_cast<uintptr_t>(data) + offset;
    const T* ptr = reinterpret_cast<const T*>(addr);
    value = *ptr;
    offset += sizeof(T);
}

inline void trt_deserialize_value(const void* data, size_t length, size_t& offset, void* value, size_t size)
{
    assert(offset < length);
    uintptr_t addr = reinterpret_cast<uintptr_t>(data) + offset;
    const void* ptr = reinterpret_cast<const void*>(addr);
    memcpy(value, ptr, size);
    offset += size;
}

template <typename T>
inline void trt_deserialize_value(const void* data, size_t length, size_t& offset, std::vector<T>& value)
{
    assert(offset < length);
    size_t count = 0;
    trt_deserialize_value(data, length, offset, count);
    assert(count);
    value.resize(count);
    trt_deserialize_value(data, length, offset, value.data(), count*sizeof(T));
}

inline void trt_deserialize_value(const void* data, size_t length, size_t& offset, Weights& w)
{
    assert(offset < length);
    trt_deserialize_value(data, length, offset, w.type);
    trt_deserialize_value(data, length, offset, w.count);
    assert(w.count);
    size_t size = w.count*trt_dtype_size(w.type);
    auto* ptr = malloc(size);
    trt_deserialize_value(data, length, offset, ptr, size);
    w.values = ptr;
}

inline DataType trt_field_type_to_dtype(PluginFieldType type)
{

    switch (type)
    {
        case PluginFieldType::kFLOAT32:
            return DataType::kFLOAT;
        case PluginFieldType::kFLOAT16:
            return DataType::kHALF;
        case PluginFieldType::kINT32:
            return DataType::kINT32;
        default:// should NOT be here
            assert(0);
    }
    return DataType::kFLOAT;
}

namespace plugin
{

// +------- Plugin body ----------------------------------------------------------------------------
template <typename T>
class AttentionPlugin: public IPluginV2DynamicExt
{
private:
    const std::string   name_;
    std::string         namespace_;
    cublasHandle_t      cublasHandle_;
    // scalar need copy
    struct
    {
        int batch_size      = 1;
        int batch_pos       = 1;
        int time1           = 48;
        int time2           = 48;
        int head_num        = 4;//8or4;
        int size_per_head   = 64;//64or32;
        int hidden_dim      = 0;
        int buf_size        = 0;
        int pbuf_size       = 0;
    } m_;

public:
    AttentionPlugin(const std::string& name) : name_(name)
    {
        DEBUG_FUNC();      
        cublasCreate(&cublasHandle_);
    }

    AttentionPlugin(const std::string& name, const void* data, size_t length) : name_(name)
    {
        DEBUG_FUNC();
        cublasCreate(&cublasHandle_);
        memcpy(&m_, data, sizeof(m_));
    }
    
    AttentionPlugin() = delete;

    ~AttentionPlugin()
    {
        DEBUG_FUNC();
    }

    size_t getSerializationSize() const noexcept override
    {
        DEBUG_FUNC();
        return sizeof(m_);
    }
    
    void serialize(void *buffer) const noexcept override
    {
        DEBUG_FUNC();
        memcpy(buffer, &m_, sizeof(m_));
    }
  
    IPluginV2DynamicExt* clone() const noexcept override
    {
        DEBUG_FUNC();
        auto p = new AttentionPlugin(name_);
        p->setPluginNamespace(namespace_.c_str());
        p->m_ = this->m_;
        return p;
    }

    int getNbOutputs() const noexcept override
    {
        DEBUG_FUNC();
        return 1;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        // DEBUG_FUNC();
        // return inputs[0];
        DimsExprs out;
        out.nbDims = 3;
        out.d[0] = inputs[0].d[0];;
        out.d[1] = inputs[0].d[1];
        out.d[2] =  exprBuilder.constant(256);
        return out;

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
            res = inOut[pos].type == trt_dtype<T>() && inOut[pos].dims.nbDims == 3; break;
        case 1:
            res = inOut[pos].type == trt_dtype<T>() && inOut[pos].dims.nbDims == 3; break;
        case 2:
            //res = inOut[pos].type == DataType::kINT32 && inOut[pos].dims.nbDims == 3; break;
            res = inOut[pos].type == trt_dtype<T>() && inOut[pos].dims.nbDims == 3; break;
        case 3:        
            res = inOut[pos].type == trt_dtype<T>(); break; // && inOut[pos].dims.nbDims == 2; break;
        case 4:
            res = inOut[pos].type == trt_dtype<T>(); break; // && inOut[pos].dims.nbDims == 2; break;
        case 5:
            res = inOut[pos].type == trt_dtype<T>(); break; // && inOut[pos].dims.nbDims == 2; break;
        case 6:
            res = inOut[pos].type == trt_dtype<T>(); break; // && inOut[pos].dims.nbDims == 1; break;
        case 7:
            res = inOut[pos].type == trt_dtype<T>(); break; // && inOut[pos].dims.nbDims == 2; break;
        case 8:
            res = inOut[pos].type == trt_dtype<T>(); break; // && inOut[pos].dims.nbDims == 1; break;
        case 9:
            res = inOut[pos].type == trt_dtype<T>(); break; // && inOut[pos].dims.nbDims == 2; break;
        case 10:
            res = inOut[pos].type == trt_dtype<T>(); break; // && inOut[pos].dims.nbDims == 1; break;
        case 11:
            res = inOut[pos].type == trt_dtype<T>(); break; // && inOut[pos].dims.nbDims == 2; break;
        case 12:
            res = inOut[pos].type == trt_dtype<T>(); break; // && inOut[pos].dims.nbDims == 1; break;
        default:// should NOT be here
            res = false;
        }
#if DEBUG_ENABLE        
        printf("Dim(");
        for(int i=0;i<17;i++)
        {   
            printf("%d,",inOut[i].dims.nbDims);
        }
        printf("),res(%d,%d),(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d)\n",
                pos,int(res),
                int(inOut[ 0].type),int(inOut[ 1].type),int(inOut[ 2].type),int(inOut[ 3].type),int(inOut[ 4].type),
                int(inOut[ 5].type),int(inOut[ 6].type),int(inOut[ 7].type),int(inOut[ 8].type),int(inOut[ 9].type),
                int(inOut[10].type),int(inOut[11].type),int(inOut[12].type)
              );
#endif                
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
        m_.batch_size       = in[0].max.d[0];
        //m_.batch_pos        = 1;
        m_.time1            = in[0].max.d[1];
        m_.time2            = in[0].max.d[1];
        //m_.head_num         = 4;
        //m_.size_per_head    = 32;
        m_.hidden_dim       = m_.head_num * m_.size_per_head;
        m_.buf_size         = m_.batch_size * m_.time1 * m_.hidden_dim;
        m_.pbuf_size        = m_.batch_pos * m_.time2 * m_.hidden_dim;

#if DEBUG_ENABLE
        printf("[AttentionPlugin::configurePlugin]m_:\n");
        printf("\tbatch_size = %d\n",m_.batch_size);
        printf("\tbatch_pos = %d\n",m_.batch_pos);
        printf("\ttime1 = %d\n",m_.time1);
        printf("\ttime2 = %d\n",m_.time2);
        printf("\thead_num = %d\n",m_.head_num);
        printf("\tsize_per_head = %d\n",m_.size_per_head);
        printf("\thidden_dim = %d\n",m_.hidden_dim);
        printf("\tbuf_size = %d\n",m_.buf_size);
        printf("\tpbuf_size = %d\n",m_.pbuf_size);
#endif
   
    }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,int32_t nbOutputs) const noexcept override
    {
        DEBUG_FUNC();
        size_t nElement = 0;
        int batch_size  = inputs[0].dims.d[0];
        int time1       = inputs[0].dims.d[1];
        int time2       = inputs[0].dims.d[1];
        int buf_size    = batch_size   * time1 * m_.hidden_dim;
        int pbuf_size   = m_.batch_pos * time2 * m_.hidden_dim;

        nElement += buf_size * 3 + pbuf_size;                                   // q,k,v,p
        nElement += buf_size * 4;                                               // q_transpose,k_transpose,v_transpose,p_transpose
        nElement += buf_size * 2;                                               // q_with_bias_u,q_with_bias_v
        nElement += batch_size * m_.head_num * time1 * time2 * 2;               // matrix_ac,matrix_bd
        nElement += batch_size * m_.head_num * time1 * time2 *2;                   // score score1
        nElement += batch_size * m_.head_num * time1 * m_.size_per_head * 3;    // x,x_transpose p_stack
        nElement += batch_size * m_.head_num * time1 * m_.size_per_head * 3;     // x_q_stack
        // nElement += m_.hidden_dim * m_.hidden_dim * 3;                          // qkv_weight_buf_dev_
        // nElement += m_.hidden_dim * 3;                                          // qkv_bias_buf_dev_
        return nElement * sizeof(T);
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
        cublasDestroy(cublasHandle_);
        //delete this;
    }
    
    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
}; // class AttentionPlugin

class AttentionPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection fc_;
    static std::vector<PluginField> attr_;
    std::string namespace_;

public:
    AttentionPluginCreator()
    {
        fc_.nbFields = attr_.size();
        fc_.fields = attr_.data();
    }

    ~AttentionPluginCreator() {}

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
    {
        DEBUG_FUNC();
        if(useFP16)
        {
            printf("create fp16 kernle");
            return new AttentionPlugin<half>(name);
        }
        else
        {   
            printf("create fp32 kernle");
            return new AttentionPlugin<float>(name);
        }
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        size_t offset = 0;
        auto dtype = DataType::kFLOAT;
        trt_deserialize_value(serialData, serialLength, offset, dtype);
        
        //if(dtype == nvinfer1::DataType::kHALF)
        if(useFP16)
        {
            return new AttentionPlugin<half>(name, serialData, serialLength);
        }
        else
        {
            return new AttentionPlugin<float>(name, serialData, serialLength);
        }       
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
}; // class AttentionPluginCreator

} // namespace plugin

} // namespace nvinfer1

