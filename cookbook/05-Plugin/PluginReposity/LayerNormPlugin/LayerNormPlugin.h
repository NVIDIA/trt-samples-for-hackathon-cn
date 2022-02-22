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
#include <cassert>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <string>
#include <vector>

#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))
#define CEIL_TO(X, Y)     (((X) + (Y)-1) / (Y) * (Y))

#ifdef USE_FP_16
    #if USE_FP_16 == 1
const bool useFP16 = true;
        #define EPSILON 1.0e-6f
    #else // # if USE_FP_16 == 1
const bool useFP16 = false;
        #define EPSILON 1.0e-12f
    #endif // # if USE_FP_16 == 1
#else      // # ifdef USE_FP_16
const bool useFP16 = false;
    #define EPSILON 1.0e-12f
#endif // # ifdef USE_FP_16

template<typename T>
__device__ T negtiveInfinity();

template<>
__device__ float negtiveInfinity<float>()
{
    return (float)-3.0e38;
}

template<>
__device__ half negtiveInfinity<half>()
{
    return (half)-6.0e6;
}

// +------- Debug wrapper --------------------------------------------------------------------------
#if DEBUG_ENABLE
    #define DEBUG_FUNC()                                 \
        do                                               \
        {                                                \
            printf("[%s]: this=->%p\n", __func__, this); \
        } while (0);
#else
    #define DEBUG_FUNC()
#endif // DEBUG_ENABLE

template<int NUM>
__forceinline__ __device__ __host__ int round(int num)
{
    return ((num - 1) / NUM + 1) * NUM;
}

template<typename T>
__global__ void print(const void *, bool);

template<>
__global__ void print<float>(const void *input0, bool isFP)
{
    if (isFP)
    {
        float *input = (float *)input0;
        for (int i = 0; i < 10; i++)
            printf("%.3f,", input[i]);
        printf("\n");
    }
    else
    {
        int *input = (int *)input0;
        for (int i = 0; i < 10; i++)
            printf("%2d,", input[i]);
        printf("\n");
    }
}

template<>
__global__ void print<half>(const void *input0, bool isFP)
{
    if (isFP)
    {
        half *input = (half *)input0;
        for (int i = 0; i < 10; i++)
            printf("%.3f,", __half2float(input[i]));
        printf("\n");
    }
    else
    {
        int *input = (int *)input0;
        for (int i = 0; i < 10; i++)
            printf("%2d,", input[i]);
        printf("\n");
    }
}

// +------- Plguin ---------------------------------------------------------------------------------
namespace
{
static const char *PLUGIN_NAME {"LayerNorm"};
static const char *PLUGIN_VERSION {"1"};
} // namespace

// +------- cuBLAS wrapper -------------------------------------------------------------------------
template<typename T>
cudaDataType_t cublas_dtype();

template<>
cudaDataType_t cublas_dtype<float>()
{
    return CUDA_R_32F;
}

template<>
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
template<typename T>
DataType trt_dtype();

template<>
DataType trt_dtype<float>()
{
    return DataType::kFLOAT;
}

template<>
DataType trt_dtype<half>()
{
    return DataType::kHALF;
}

int trt_dtype_size(DataType dtype)
{
    switch (dtype)
    {
    case DataType::kFLOAT:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT32:
        return 4;
    default: // should NOT be here
        assert(0);
    }
    return 0;
}

template<typename T>
inline size_t trt_serialize_size(T value)
{
    return sizeof(value);
}

template<typename T>
inline size_t trt_serialize_size(const std::vector<T> &value)
{
    return sizeof(value.size()) + value.size() * sizeof(T);
}

inline size_t trt_serialize_size(const Weights &w)
{
    return sizeof(w.type) + sizeof(w.count) + w.count * trt_dtype_size(w.type);
}

template<typename T>
inline void trt_serialize_value(void **buffer, T value)
{
    T *ptr         = reinterpret_cast<T *>(*buffer);
    *ptr           = value;
    uintptr_t addr = reinterpret_cast<uintptr_t>(*buffer);
    addr += sizeof(T);
    *buffer = reinterpret_cast<void *>(addr);
}

inline void trt_serialize_value(void **buffer, const void *value, size_t size)
{
    void *ptr = *buffer;
    memcpy(ptr, value, size);
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    addr += size;
    *buffer = reinterpret_cast<void *>(addr);
}

template<typename T>
inline void trt_serialize_value(void **buffer, const std::vector<T> &value)
{
    trt_serialize_value(buffer, value.size());
    size_t size = value.size() * sizeof(T);
    trt_serialize_value(buffer, value.data(), size);
}

inline void trt_serialize_value(void **buffer, const Weights &w)
{
    trt_serialize_value(buffer, w.type);
    trt_serialize_value(buffer, w.count);
    size_t size = w.count * trt_dtype_size(w.type);
    trt_serialize_value(buffer, w.values, size);
}

template<typename T>
inline void trt_deserialize_value(const void *data, size_t length, size_t &offset, T &value)
{
    assert(offset < length);
    uintptr_t addr = reinterpret_cast<uintptr_t>(data) + offset;
    const T * ptr  = reinterpret_cast<const T *>(addr);
    value          = *ptr;
    offset += sizeof(T);
}

inline void trt_deserialize_value(const void *data, size_t length, size_t &offset, void *value, size_t size)
{
    assert(offset < length);
    uintptr_t   addr = reinterpret_cast<uintptr_t>(data) + offset;
    const void *ptr  = reinterpret_cast<const void *>(addr);
    memcpy(value, ptr, size);
    offset += size;
}

template<typename T>
inline void trt_deserialize_value(const void *data, size_t length, size_t &offset, std::vector<T> &value)
{
    assert(offset < length);
    size_t count = 0;
    trt_deserialize_value(data, length, offset, count);
    assert(count);
    value.resize(count);
    trt_deserialize_value(data, length, offset, value.data(), count * sizeof(T));
}

inline void trt_deserialize_value(const void *data, size_t length, size_t &offset, Weights &w)
{
    assert(offset < length);
    trt_deserialize_value(data, length, offset, w.type);
    trt_deserialize_value(data, length, offset, w.count);
    assert(w.count);
    size_t size = w.count * trt_dtype_size(w.type);
    auto * ptr  = malloc(size);
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
    default: // should NOT be here
        assert(0);
    }
    return DataType::kFLOAT;
}

namespace plugin
{
// +------- Plugin body ----------------------------------------------------------------------------
template<typename T>
class LayerNormPlugin : public IPluginV2DynamicExt
{
private:
    std::string name_;
    std::string namespace_;

public:
    LayerNormPlugin(const std::string &name):
        name_(name)
    {
        DEBUG_FUNC();
    }

    LayerNormPlugin(const std::string &name, const void *data, size_t length):
        name_(name)
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

    IPluginV2DynamicExt *clone() const noexcept override
    {
        DEBUG_FUNC();
        return new LayerNormPlugin<T>(name_);
    }

    int getNbOutputs() const noexcept override
    {
        DEBUG_FUNC();
        return 1;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override
    {
        DEBUG_FUNC();
        // old version: (-1,-1,560/320) -> (-1,-1,560/320)
        //return inputs[0];
        // new version: (-1,-1,560/320) -> (-1,1,-1,560/320)
        DimsExprs out;
        out.nbDims = 4;
        out.d[0]   = inputs[0].d[0];
        out.d[1]   = exprBuilder.constant(1);
        out.d[2]   = inputs[0].d[1];
        out.d[3]   = inputs[0].d[2];
        return out;
    }

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        DEBUG_FUNC();
        if (inOut[pos].format != TensorFormat::kLINEAR)
        {
            return false;
        }

        bool res = false;
        switch (pos)
        {
        case 0:
            res = inOut[pos].type == trt_dtype<T>() && inOut[pos].dims.nbDims == 3;
            break;
        case 1:
            res = inOut[pos].type == DataType::kFLOAT && inOut[pos].dims.nbDims == 1;
            break;
        case 2:
            res = inOut[pos].type == DataType::kFLOAT && inOut[pos].dims.nbDims == 1;
            break;
        case 3:
            res = inOut[pos].type == trt_dtype<T>() && inOut[pos].dims.nbDims == 4;
            break;
        default: // should NOT be here
            break;
        }
#if DEBUG_ENABLE
        printf("Dim(");
        for (int i = 0; i < 2; i++)
        {
            printf("%d,", inOut[i].dims.nbDims);
        }
        printf("),res(%d,%d),(%d,%d,%d,%d)\n", pos, int(res), int(inOut[0].type), int(inOut[1].type), int(inOut[2].type), int(inOut[3].type));
#endif
        return res;
    }

    DataType getOutputDataType(int outputIndex, const DataType *inputTypes, int nbInputs) const noexcept override
    {
        DEBUG_FUNC();
        return inputTypes[0];
    }

    void configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override
    {
        DEBUG_FUNC();
#if DEBUG_ENABLE
        printf("[LayerNormPlugin::configurePlugin]\n");
#endif
    }

    size_t getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override
    {
        DEBUG_FUNC();
        return 0;
    }

    void setPluginNamespace(const char *szNamespace) noexcept override
    {
        DEBUG_FUNC();
        namespace_ = szNamespace;
    }
    const char *getPluginNamespace() const noexcept override
    {
        DEBUG_FUNC();
        return namespace_.c_str();
    }
    const char *getPluginType() const noexcept override
    {
        DEBUG_FUNC();
        return PLUGIN_NAME;
    }
    const char *getPluginVersion() const noexcept override
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

    int32_t enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
}; // class LayerNormPlugin

class LayerNormPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    LayerNormPluginCreator()
    {
        fc_.nbFields = attr_.size();
        fc_.fields   = attr_.data();
    }

    ~LayerNormPluginCreator() {}

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
    {
        DEBUG_FUNC();
        if (useFP16)
        {
            return new LayerNormPlugin<half>(name);
        }
        else
        {
            return new LayerNormPlugin<float>(name);
        }
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
    {
        size_t offset = 0;
        auto   dtype  = DataType::kFLOAT;
        trt_deserialize_value(serialData, serialLength, offset, dtype);

        if (useFP16)
        {
            return new LayerNormPlugin<half>(name, serialData, serialLength);
        }
        else
        {
            return new LayerNormPlugin<float>(name, serialData, serialLength);
        }
    }

    void setPluginNamespace(const char *szNamespace) noexcept override
    {
        namespace_ = szNamespace;
    }

    const char *getPluginNamespace() const noexcept override
    {
        return namespace_.c_str();
    }

    const char *getPluginName() const noexcept override
    {
        return PLUGIN_NAME;
    }

    const char *getPluginVersion() const noexcept override
    {
        return PLUGIN_VERSION;
    }

    const PluginFieldCollection *getFieldNames() noexcept override
    {
        return &fc_;
    }
}; // class LayerNormPluginCreator

} // namespace plugin

} // namespace nvinfer1
