/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

/// Debug wrapper ///
inline void check(cublasStatus_t ret, int line)
{
    if (ret != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS Error: " << ret << ", line: " << line << std::endl;
    }
}

inline void check(cudaError_t ret, int line)
{
    if (ret != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(ret) << ", line: " << line << std::endl;
    }
}

#define CHECK(_x) check((_x), __LINE__)

#if DEBUG_ENABLE
    #define DEBUG_FUNC()                                                        \
        do                                                                      \
        {                                                                       \
            std::cout << "\tDEBUG:\t" << this << "\t" << __func__ << std::endl; \
        } while (0)
#else
    #define DEBUG_FUNC()
#endif

/// TRT wrapper ///
template<typename T>
nvinfer1::DataType trt_dtype();

template<>
nvinfer1::DataType trt_dtype<float>()
{
    return nvinfer1::DataType::kFLOAT;
}

template<>
nvinfer1::DataType trt_dtype<half>()
{
    return nvinfer1::DataType::kHALF;
}

int trt_dtype_size(nvinfer1::DataType dtype)
{
    if (dtype == nvinfer1::DataType::kFLOAT)
        return 4;
    if (dtype == nvinfer1::DataType::kHALF)
        return 2;
    assert(0); // should NOT be here!
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

inline size_t trt_serialize_size(const nvinfer1::Weights &w)
{
    return sizeof(w.type) + sizeof(w.count) +
           w.count * trt_dtype_size(w.type);
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

inline void trt_serialize_value(void **buffer, const nvinfer1::Weights &w)
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

inline void trt_deserialize_value(const void *data, size_t length, size_t &offset, nvinfer1::Weights &w)
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

inline nvinfer1::DataType trt_field_type_to_dtype(nvinfer1::PluginFieldType type)
{
    switch (type)
    {
    case nvinfer1::PluginFieldType::kFLOAT32:
    {
        return nvinfer1::DataType::kFLOAT;
    }
    case nvinfer1::PluginFieldType::kFLOAT16:
    {
        return nvinfer1::DataType::kHALF;
    }
    case nvinfer1::PluginFieldType::kINT32:
    {
        return nvinfer1::DataType::kINT32;
    }
    case nvinfer1::PluginFieldType::kINT8:
    {
        return nvinfer1::DataType::kINT8;
    }
    default:
    {
        // should NOT be here!
        assert(0);
    }
    }
    return nvinfer1::DataType::kFLOAT;
}

/// cuBLAS wrapper ///
int cublas_dtype_size(cudaDataType_t dtype)
{
    if (dtype == CUDA_R_32F)
        return 4;
    if (dtype == CUDA_R_16F)
        return 2;
    assert(0); // should NOT be here!
    return 0;
}

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

cublasStatus_t row_major_gemm(cublasHandle_t    handle,
                              cublasOperation_t transa,
                              cublasOperation_t transb,
                              int               m,
                              int               n,
                              int               k,
                              const void *      alpha,
                              const void *      A,
                              cudaDataType_t    Atype,
                              int               lda,
                              const void *      B,
                              cudaDataType_t    Btype,
                              int               ldb,
                              const void *      beta,
                              void *            C,
                              cudaDataType_t    Ctype,
                              int               ldc,
                              cudaDataType      computeType, //cublasComputeType_t computeType,
                              cublasGemmAlgo_t  algo)
{
    return cublasGemmEx(handle,
                        transb,
                        transa,
                        n,
                        m,
                        k,
                        alpha,
                        B,
                        Btype,
                        ldb,
                        A,
                        Atype,
                        lda,
                        beta,
                        C,
                        Ctype,
                        ldc,
                        computeType,
                        algo);
}

cublasStatus_t row_major_stride_batched_gemm(cublasHandle_t    handle,
                                             cublasOperation_t transa,
                                             cublasOperation_t transb,
                                             int               m,
                                             int               n,
                                             int               k,
                                             const void *      alpha,
                                             const void *      A,
                                             cudaDataType_t    Atype,
                                             int               lda,
                                             long long int     strideA,
                                             const void *      B,
                                             cudaDataType_t    Btype,
                                             int               ldb,
                                             long long int     strideB,
                                             const void *      beta,
                                             void *            C,
                                             cudaDataType_t    Ctype,
                                             int               ldc,
                                             long long int     strideC,
                                             int               batchCount,
                                             cudaDataType      computeType, //cublasComputeType_t computeType,
                                             cublasGemmAlgo_t  algo)
{
    return cublasGemmStridedBatchedEx(handle,
                                      transb,
                                      transa,
                                      n,
                                      m,
                                      k,
                                      alpha,
                                      B,
                                      Btype,
                                      ldb,
                                      strideB,
                                      A,
                                      Atype,
                                      lda,
                                      strideA,
                                      beta,
                                      C,
                                      Ctype,
                                      ldc,
                                      strideC,
                                      batchCount,
                                      computeType,
                                      algo);
}

cublasStatus_t row_major_batched_gemm(cublasHandle_t    handle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int               m,
                                      int               n,
                                      int               k,
                                      const void *      alpha,
                                      const void *      Aarray[],
                                      cudaDataType      Atype,
                                      int               lda,
                                      const void *      Barray[],
                                      cudaDataType      Btype,
                                      int               ldb,
                                      const void *      beta,
                                      void *            Carray[],
                                      cudaDataType      Ctype,
                                      int               ldc,
                                      int               batchCount,
                                      cudaDataType      computeType, //cublasComputeType_t computeType,
                                      cublasGemmAlgo_t  algo)
{
    return cublasGemmBatchedEx(handle,
                               transb,
                               transa,
                               n,
                               m,
                               k,
                               alpha,
                               Barray,
                               Btype,
                               ldb,
                               Aarray,
                               Atype,
                               lda,
                               beta,
                               Carray,
                               Ctype,
                               ldc,
                               batchCount,
                               computeType,
                               algo);
}

namespace
{
const cublasGemmAlgo_t kGemmAlgo[] = {
    CUBLAS_GEMM_DEFAULT,
    CUBLAS_GEMM_ALGO0,
    CUBLAS_GEMM_ALGO1,
    CUBLAS_GEMM_ALGO2,
    CUBLAS_GEMM_ALGO3,
    CUBLAS_GEMM_ALGO4,
    CUBLAS_GEMM_ALGO5,
    CUBLAS_GEMM_ALGO6,
    CUBLAS_GEMM_ALGO7,
    CUBLAS_GEMM_ALGO8,
    CUBLAS_GEMM_ALGO9,
    CUBLAS_GEMM_ALGO10,
    CUBLAS_GEMM_ALGO11,
    CUBLAS_GEMM_ALGO12,
    CUBLAS_GEMM_ALGO13,
    CUBLAS_GEMM_ALGO14,
    CUBLAS_GEMM_ALGO15,
    CUBLAS_GEMM_ALGO16,
    CUBLAS_GEMM_ALGO17,
    CUBLAS_GEMM_ALGO18,
    CUBLAS_GEMM_ALGO19,
    CUBLAS_GEMM_ALGO20,
    CUBLAS_GEMM_ALGO21,
    CUBLAS_GEMM_ALGO22,
    CUBLAS_GEMM_ALGO23,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    CUBLAS_GEMM_ALGO0_TENSOR_OP,
    CUBLAS_GEMM_ALGO1_TENSOR_OP,
    CUBLAS_GEMM_ALGO2_TENSOR_OP,
    CUBLAS_GEMM_ALGO3_TENSOR_OP,
    CUBLAS_GEMM_ALGO4_TENSOR_OP,
    CUBLAS_GEMM_ALGO5_TENSOR_OP,
    CUBLAS_GEMM_ALGO6_TENSOR_OP,
    CUBLAS_GEMM_ALGO7_TENSOR_OP,
    CUBLAS_GEMM_ALGO8_TENSOR_OP,
    CUBLAS_GEMM_ALGO9_TENSOR_OP,
    CUBLAS_GEMM_ALGO10_TENSOR_OP,
    CUBLAS_GEMM_ALGO11_TENSOR_OP,
    CUBLAS_GEMM_ALGO12_TENSOR_OP,
    CUBLAS_GEMM_ALGO13_TENSOR_OP,
    CUBLAS_GEMM_ALGO14_TENSOR_OP,
    CUBLAS_GEMM_ALGO15_TENSOR_OP,
};

const int              kInvalidAlgoIdx = -99;
const cublasGemmAlgo_t kInvalidAglo    = static_cast<cublasGemmAlgo_t>(-99);
} // namespace

template<typename T>
cublasGemmAlgo_t find_fastest_gemm_algo(cublasHandle_t    handle,
                                        cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        int               m,
                                        int               n,
                                        int               k)
{
    T *A, *B, *C;
    CHECK(cudaMalloc((void **)&A, m * k * cublas_dtype_size(cublas_dtype<T>())));
    CHECK(cudaMalloc((void **)&B, k * n * cublas_dtype_size(cublas_dtype<T>())));
    CHECK(cudaMalloc((void **)&C, m * n * cublas_dtype_size(cublas_dtype<T>())));

    int lda = (transa == CUBLAS_OP_N) ? k : m;
    int ldb = (transb == CUBLAS_OP_N) ? n : k;
    int ldc = n;

    float alpha = 1.f;
    float beta  = 0.f;

    int         num_algo = sizeof(kGemmAlgo) / sizeof(cublasGemmAlgo_t);
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    int iter = 100;

    float min_time    = FLT_MAX;
    int   fastest_idx = kInvalidAlgoIdx;
    for (int i = 0; i < num_algo; ++i)
    {
        auto algo = kGemmAlgo[i];

        // warmup
        auto status = row_major_gemm(handle,
                                     transa,
                                     transb,
                                     m,
                                     n,
                                     k,
                                     &alpha,
                                     reinterpret_cast<const void *>(A),
                                     cublas_dtype<T>(),
                                     lda,
                                     reinterpret_cast<const void *>(B),
                                     cublas_dtype<T>(),
                                     ldb,
                                     &beta,
                                     C,
                                     cublas_dtype<T>(),
                                     ldc,
                                     CUDA_R_32F, //CUBLAS_COMPUTE_32F,
                                     algo);

        if (status != CUBLAS_STATUS_SUCCESS)
        {
#if DEBUG_ENABLE
            std::cerr << "cuBLAS ERROR: " << status << ", algo: " << algo << std::endl;
#endif
            continue;
        }

        CHECK(cudaEventRecord(start));
        for (int j = 0; j < iter; ++j)
        {
            row_major_gemm(handle,
                           transa,
                           transb,
                           m,
                           n,
                           k,
                           &alpha,
                           reinterpret_cast<const void *>(A),
                           cublas_dtype<T>(),
                           lda,
                           reinterpret_cast<const void *>(B),
                           cublas_dtype<T>(),
                           ldb,
                           &beta,
                           C,
                           cublas_dtype<T>(),
                           ldc,
                           CUDA_R_32F, //CUBLAS_COMPUTE_32F,
                           algo);
        }
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float tmp;
        CHECK(cudaEventElapsedTime(&tmp, start, stop));
#if DEBUG_ENABLE
        std::cout << "algo: " << algo << ", time: " << tmp << ", min: " << min_time << std::endl;
#endif
        if (tmp < min_time)
        {
            min_time    = tmp;
            fastest_idx = i;
        }
    }

    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(C));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
#if DEBUG_ENABLE
    std::cout << "cuBLAS fastest gemm algo: " << kGemmAlgo[fastest_idx] << std::endl;
#endif
    return kGemmAlgo[fastest_idx];
}

template<typename T>
cublasGemmAlgo_t find_fastest_batched_gemm_algo(cublasHandle_t    handle,
                                                cublasOperation_t transa,
                                                cublasOperation_t transb,
                                                int               m,
                                                int               n,
                                                int               k,
                                                int               batchCount)
{
    T *    A, *B, *C;
    void **Aarray, **Barray, **Carray;
    CHECK(cudaMalloc((void **)&A, m * k * batchCount * cublas_dtype_size(cublas_dtype<T>())));
    CHECK(cudaMalloc((void **)&B, k * n * batchCount * cublas_dtype_size(cublas_dtype<T>())));
    CHECK(cudaMalloc((void **)&C, m * n * batchCount * cublas_dtype_size(cublas_dtype<T>())));
    CHECK(cudaMallocManaged((void **)&Aarray, batchCount * sizeof(void *)));
    CHECK(cudaMallocManaged((void **)&Barray, batchCount * sizeof(void *)));
    CHECK(cudaMallocManaged((void **)&Carray, batchCount * sizeof(void *)));

    for (int i = 0; i < batchCount; ++i)
    {
        Aarray[i] = reinterpret_cast<void *>(A + i * (m * k));
        Barray[i] = reinterpret_cast<void *>(B + i * (k * n));
        Carray[i] = reinterpret_cast<void *>(C + i * (m * n));
    }

    int lda = (transa == CUBLAS_OP_N) ? k : m;
    int ldb = (transb == CUBLAS_OP_N) ? n : k;
    int ldc = n;

    float alpha = 1.f;
    float beta  = 0.f;

    int         num_algo = sizeof(kGemmAlgo) / sizeof(cublasGemmAlgo_t);
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    int iter = 100;

    float min_time    = FLT_MAX;
    int   fastest_idx = kInvalidAlgoIdx;
    for (int i = 0; i < num_algo; ++i)
    {
        auto algo = kGemmAlgo[i];

        // warmup
        auto status = row_major_batched_gemm(handle,
                                             transa,
                                             transb,
                                             m,
                                             n,
                                             k,
                                             &alpha,
                                             const_cast<const void **>(Aarray),
                                             cublas_dtype<T>(),
                                             lda,
                                             const_cast<const void **>(Barray),
                                             cublas_dtype<T>(),
                                             ldb,
                                             &beta,
                                             Carray,
                                             cublas_dtype<T>(),
                                             ldc,
                                             batchCount,
                                             CUDA_R_32F, //CUBLAS_COMPUTE_32F,
                                             algo);

        if (status != CUBLAS_STATUS_SUCCESS)
        {
#if DEBUG_ENABLE
            std::cerr << "cuBLAS ERROR: " << status << ", algo: " << algo << std::endl;
#endif
            continue;
        }

        CHECK(cudaEventRecord(start));
        for (int j = 0; j < iter; ++j)
        {
            row_major_batched_gemm(handle,
                                   transa,
                                   transb,
                                   m,
                                   n,
                                   k,
                                   &alpha,
                                   const_cast<const void **>(Aarray),
                                   cublas_dtype<T>(),
                                   lda,
                                   const_cast<const void **>(Barray),
                                   cublas_dtype<T>(),
                                   ldb,
                                   &beta,
                                   Carray,
                                   cublas_dtype<T>(),
                                   ldc,
                                   batchCount,
                                   CUDA_R_32F, //CUBLAS_COMPUTE_32F,
                                   algo);
        }
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float tmp;
        CHECK(cudaEventElapsedTime(&tmp, start, stop));
#if DEBUG_ENABLE
        std::cout << "algo: " << algo << ", time: " << tmp << ", min: " << min_time << std::endl;
#endif
        if (tmp < min_time)
        {
            min_time    = tmp;
            fastest_idx = i;
        }
    }

    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(C));
    CHECK(cudaFree(Aarray));
    CHECK(cudaFree(Barray));
    CHECK(cudaFree(Carray));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
#if DEBUG_ENABLE
    std::cout << "cuBLAS fastest batched gemm algo: " << kGemmAlgo[fastest_idx] << std::endl;
#endif
    return kGemmAlgo[fastest_idx];
}

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg)
    {
        // suppress info-level messages
#if DEBUG_ENABLE
        std::cout << msg << std::endl;
#endif
    }
} gLogger;

namespace
{
static const char *MATCH_MATRIX_TENSOR_VERSION {"1"};
static const char *MATCH_MATRIX_TENSOR_NAME {"MMTPlugin"};
} // namespace

template<typename T>
class MatchMatrixTensor : public nvinfer1::IPluginV2DynamicExt
{
public:
    MatchMatrixTensor(const std::string &name, nvinfer1::Weights w, int h, int dim_t, bool deep_copy_weight = false):
        name_(name), w_(w), h_(h), dim_t_(dim_t)
    {
        DEBUG_FUNC();
        assert(h > 0);
        assert(dim_t > 0);
        assert(w.type == trt_dtype<T>());
        size_t count = h * dim_t * h;
        assert(count == w.count);

        CHECK(cublasCreate(&handle_));
        CHECK(cudaStreamCreate(&copy_stream_));
        CHECK(cudaEventCreateWithFlags(&copy_event_, cudaEventDisableTiming));

        if (deep_copy_weight)
        {
virtual bool 	supportsFormat (DataType type, PluginFormat format) const noexcept=0
 	Check format support. More...
 
virtual void 	configureWithFormat (Dims const *inputDims, int32_t nbInputs, Dims const *outputDims, int32_t nbOutputs, DataType type, PluginFormat format, int32_t maxBatchSize) noexcept=0
 	Configure the layer. More...
 
virtual int32_t 	initialize () noexcept=0
 	Initialize the layer for execution. This is called when the engine is created. More...    own_w_      = true;
size_t              size = w.count * trt_dtype_size(w.type);
w_.values                = malloc(size);
memcpy(const_cast<void *>(w_.values), w.values, size);
        }
    }

    MatchMatrixTensor(const std::string &name, const void *buffer, size_t length):
        name_(name), own_w_(true)
    {
        // deserialization
        DEBUG_FUNC();

        size_t offset = 0;
        trt_deserialize_value(buffer, length, offset, w_);
        trt_deserialize_value(buffer, length, offset, h_);
        trt_deserialize_value(buffer, length, offset, dim_t_);
        trt_deserialize_value(buffer, length, offset, gemm_algo_);
        trt_deserialize_value(buffer, length, offset, batched_gemm_algo_);

        assert(h_ > 0);
        assert(dim_t_ > 0);

        CHECK(cublasCreate(&handle_));
        CHECK(cudaStreamCreate(&copy_stream_));
        CHECK(cudaEventCreateWithFlags(&copy_event_, cudaEventDisableTiming));
    }

    MatchMatrixTensor() = delete;
    ~MatchMatrixTensor()
    {
        DEBUG_FUNC();
    }

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt *clone() const
    {
        DEBUG_FUNC();
        auto p = new MatchMatrixTensor(name_, w_, h_, dim_t_);
        p->setPluginNamespace(namespace_.c_str());
        p->gemm_algo_         = this->gemm_algo_;
        p->batched_gemm_algo_ = this->batched_gemm_algo_;
        p->dev_w_             = this->dev_w_;
        return p;
    }

    nvinfer1::DimsExprs getOutputDimensions(
        int                        outputIndex,
        const nvinfer1::DimsExprs *inputs,
        int                        nbInputs,
        nvinfer1::IExprBuilder &   exprBuilder)
    {
        DEBUG_FUNC();
        nvinfer1::DimsExprs ret;
        assert(outputIndex == 0);
        ret.nbDims = 4;
        ret.d[0]   = inputs[0].d[0];               // num_seq
        ret.d[1]   = exprBuilder.constant(dim_t_); // output_channel
        ret.d[2]   = inputs[0].d[1];               // x_seq_len
        ret.d[3]   = inputs[1].d[1];               // y_seq_len
        return ret;
    }

    bool supportsFormatCombination(
        int                               pos,
        const nvinfer1::PluginTensorDesc *inOut,
        int                               nbInputs,
        int                               nbOutputs)
    {
        DEBUG_FUNC();
        assert(nbInputs == 2);
        assert(nbOutputs == 1);

        const auto &desc = inOut[pos];
        if (desc.format != nvinfer1::TensorFormat::kLINEAR)
        {
            return false;
        }
        // input
        // 0 x
        // 1 y
        if (pos < 2)
        {
            return desc.type == trt_dtype<T>() && desc.dims.nbDims == 3;
        }
        // output
        // 0 out
        return desc.type == trt_dtype<T>() && desc.dims.nbDims == 4;
    }

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs)
    {
        DEBUG_FUNC();
        assert(nbInputs == 2);
        assert(nbOutputs == 1);

        for (auto i = 0; i < 2; ++i)
        { // x, y
            const auto &dynamic_desc = in[i];
            assert(dynamic_desc.desc.dims.nbDims == 3);
            assert(dynamic_desc.desc.type == trt_dtype<T>());
            assert(dynamic_desc.desc.format == nvinfer1::TensorFormat::kLINEAR);
        }

        auto max_num_seq    = in[0].max.d[0];
        auto max_gemm_count = max_num_seq * dim_t_;
        CHECK(cudaMallocHost((void **)&host_a_array_, max_gemm_count * sizeof(void *)));
        CHECK(cudaMallocHost((void **)&host_b_array_, max_gemm_count * sizeof(void *)));
        CHECK(cudaMallocHost((void **)&host_c_array_, max_gemm_count * sizeof(void *)));
        CHECK(cudaMalloc((void **)&dev_a_array_, max_gemm_count * sizeof(void *)));
        CHECK(cudaMalloc((void **)&dev_b_array_, max_gemm_count * sizeof(void *)));
        CHECK(cudaMalloc((void **)&dev_c_array_, max_gemm_count * sizeof(void *)));

        if (gemm_algo_ == kInvalidAglo)
        {
            gemm_algo_ = find_fastest_gemm_algo<T>(handle_,
                                                   CUBLAS_OP_N,
                                                   CUBLAS_OP_N,
                                                   in[0].max.d[0] * in[0].max.d[1], // m
                                                   dim_t_ * in[0].max.d[2],         // n
                                                   in[0].max.d[2]);                 // k
        }

        if (batched_gemm_algo_ == kInvalidAglo)
        {
            batched_gemm_algo_ = find_fastest_batched_gemm_algo<T>(handle_,
                                                                   CUBLAS_OP_N,
                                                                   CUBLAS_OP_T,
                                                                   in[0].max.d[1], // m
                                                                   in[1].max.d[1], // n
                                                                   in[0].max.d[2], // k
                                                                   max_gemm_count);
        }

        { // out
            const auto &dynamic_desc = out[0];
            assert(dynamic_desc.desc.dims.nbDims == 4);
            assert(dynamic_desc.desc.type == trt_dtype<T>());
            assert(dynamic_desc.desc.format == nvinfer1::TensorFormat::kLINEAR);
        }
    }

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
    {
        DEBUG_FUNC();
        assert(inputs[0].dims.nbDims == 3);
        return (dim_t_ * inputs[0].dims.d[2]) *
               (inputs[0].dims.d[0] * inputs[0].dims.d[1]) * sizeof(T);
    }

    int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream)
    {
        DEBUG_FUNC();
        auto dim_in_ = inputDesc[0].dims.d[2]; // emb_size
        assert(dim_in_ == h_);

        auto num_seq = inputDesc[0].dims.d[0];

        size_t                   lod_size = num_seq + 1;
        thrust::host_vector<int> offset_l {lod_size};
        thrust::host_vector<int> offset_r {lod_size};
        thrust::host_vector<int> top_offset {lod_size};

        offset_l[0]    = 0;
        offset_r[0]    = 0;
        auto x_seq_len = inputDesc[0].dims.d[1];
        auto y_seq_len = inputDesc[1].dims.d[1];

        int top_size  = 0;
        top_offset[0] = top_size;
        for (size_t b = 0; b < lod_size - 1; b++)
        {
            offset_l[b + 1] = offset_l[b] + x_seq_len;
            offset_r[b + 1] = offset_r[b] + y_seq_len;
            top_size += dim_t_ * x_seq_len * y_seq_len;
            top_offset[b + 1] = top_size;
        }

        auto *bottom_l_data       = reinterpret_cast<const T *>(inputs[0]);
        auto *bottom_r_data       = reinterpret_cast<const T *>(inputs[1]);
        auto *t_data              = dev_w_;
        auto *out_data            = reinterpret_cast<T *>(outputs[0]);
        auto *bottom_l_trans_data = reinterpret_cast<T *>(workspace);

        float alpha = 1.f, beta = 0.f;

        CHECK(cublasSetStream(handle_, stream));
        CHECK(row_major_gemm(handle_,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             num_seq * x_seq_len, // m
                             dim_t_ * dim_in_,    // n
                             dim_in_,             // k
                             &alpha,              // alpha
                             bottom_l_data,       // A
                             cublas_dtype<T>(),   // Atype
                             dim_in_,             // lda
                             t_data,              // B
                             cublas_dtype<T>(),   // Btype
                             dim_t_ * dim_in_,    // ldb
                             &beta,               // beta
                             bottom_l_trans_data, // C
                             cublas_dtype<T>(),   // Ctype
                             dim_t_ * dim_in_,    // ldc
                             CUDA_R_32F,          //CUBLAS_COMPUTE_32F,
                             gemm_algo_));

        int len_l = offset_l[1] - offset_l[0];
        int len_r = offset_r[1] - offset_r[0];

        int offset = 0;
        for (size_t b = 0; b < lod_size - 1; b++)
        {
            auto *top_data = out_data + top_offset[b];
            auto  stride_c = len_l * len_r;
            for (int i = 0; i < dim_t_; ++i)
            {
                host_c_array_[offset + i] = const_cast<T *>(top_data);
                top_data += stride_c;
            }

            const auto *l_t_data = bottom_l_trans_data + offset_l[b] * dim_t_ * dim_in_;
            auto        stride_a = dim_in_;
            for (int i = 0; i < dim_t_; ++i)
            {
                host_a_array_[offset + i] = const_cast<T *>(l_t_data);
                l_t_data += stride_a;
            }

            const auto *r_data = bottom_r_data + offset_r[b] * dim_in_;
            //auto stride_b = 0;
            for (int i = 0; i < dim_t_; ++i)
            {
                host_b_array_[offset + i] = const_cast<T *>(r_data);
            }

            offset += dim_t_;
        }

        CHECK(cudaMemcpyAsync(dev_a_array_, host_a_array_, offset * sizeof(void *), cudaMemcpyHostToDevice, copy_stream_));
        CHECK(cudaMemcpyAsync(dev_b_array_, host_b_array_, offset * sizeof(void *), cudaMemcpyHostToDevice, copy_stream_));
        CHECK(cudaMemcpyAsync(dev_c_array_, host_c_array_, offset * sizeof(void *), cudaMemcpyHostToDevice, copy_stream_));
        CHECK(cudaEventRecord(copy_event_, copy_stream_));
        CHECK(cudaStreamWaitEvent(stream, copy_event_, 0));

        CHECK(row_major_batched_gemm(handle_,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_T,
                                     len_l,                                                              // m
                                     len_r,                                                              // n
                                     dim_in_,                                                            // k
                                     &alpha,                                                             // alpha
                                     const_cast<const void **>(reinterpret_cast<void **>(dev_a_array_)), // A
                                     cublas_dtype<T>(),                                                  // Atype
                                     dim_t_ * dim_in_,                                                   // lda
                                     const_cast<const void **>(reinterpret_cast<void **>(dev_b_array_)), // B
                                     cublas_dtype<T>(),                                                  // Btype
                                     dim_in_,                                                            // ldb
                                     &beta,                                                              // beta
                                     reinterpret_cast<void **>(dev_c_array_),                            // C
                                     cublas_dtype<T>(),                                                  // Ctype
                                     len_r,                                                              // ldc
                                     dim_t_ * num_seq,                                                   // batchCount
                                     CUDA_R_32F,                                                         //CUBLAS_COMPUTE_32F,
                                     batched_gemm_algo_));
        return 0;
    }

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
    {
        DEBUG_FUNC();
        if (index == 0)
            return trt_dtype<T>();
        return nvinfer1::DataType::kINT32;
    }

    // IPluginV2 Methods
    const char *getPluginType() const
    {
        DEBUG_FUNC();
        return MATCH_MATRIX_TENSOR_NAME;
    }
    const char *getPluginVersion() const
    {
        DEBUG_FUNC();
        return MATCH_MATRIX_TENSOR_VERSION;
    }
    int getNbOutputs() const
    {
        DEBUG_FUNC();
        return 1;
    }
    int initialize()
    {
        DEBUG_FUNC();
        CHECK(cudaMalloc((void **)&dev_w_, w_.count * sizeof(T)));
        CHECK(cudaMemcpy(dev_w_, w_.values, w_.count * sizeof(T), cudaMemcpyHostToDevice));
        return 0;
    }
    void terminate()
    {
        DEBUG_FUNC();
        CHECK(cudaFree(dev_w_));
        if (own_w_)
        {
            free(const_cast<void *>(w_.values));
        }
    }
    size_t getSerializationSize() const
    {
        DEBUG_FUNC();
        return trt_serialize_size(w_) +
               trt_serialize_size(h_) +
               trt_serialize_size(dim_t_) +
               trt_serialize_size(gemm_algo_) +
               trt_serialize_size(batched_gemm_algo_);
    }
    void serialize(void *buffer) const
    {
        DEBUG_FUNC();
        trt_serialize_value(&buffer, w_);
        trt_serialize_value(&buffer, h_);
        trt_serialize_value(&buffer, dim_t_);
        trt_serialize_value(&buffer, gemm_algo_);
        trt_serialize_value(&buffer, batched_gemm_algo_);
    }
    void destroy()
    {
        DEBUG_FUNC();
        CHECK(cudaFreeHost(host_a_array_));
        CHECK(cudaFreeHost(host_b_array_));
        CHECK(cudaFreeHost(host_c_array_));
        CHECK(cudaFree(dev_a_array_));
        CHECK(cudaFree(dev_b_array_));
        CHECK(cudaFree(dev_c_array_));
        CHECK(cublasDestroy(handle_));
        CHECK(cudaStreamDestroy(copy_stream_));
        CHECK(cudaEventDestroy(copy_event_));
    }
    void setPluginNamespace(const char *pluginNamespace)
    {
        DEBUG_FUNC();
        namespace_ = pluginNamespace;
    }
    const char *getPluginNamespace() const
    {
        DEBUG_FUNC();
        return namespace_.c_str();
    }

private:
    const std::string name_;
    nvinfer1::Weights w_ {nvinfer1::DataType::kFLOAT, nullptr, 0};
    bool              own_w_ = false;
    int               h_     = 0;
    int               dim_t_ = 0;
    std::string       namespace_;
    // dev
    cublasHandle_t   handle_            = nullptr;
    cublasGemmAlgo_t gemm_algo_         = kInvalidAglo;
    cublasGemmAlgo_t batched_gemm_algo_ = kInvalidAglo;
    T *              dev_w_             = nullptr;
    T **             dev_a_array_       = nullptr;
    T **             dev_b_array_       = nullptr;
    T **             dev_c_array_       = nullptr;
    T **             host_a_array_      = nullptr;
    T **             host_b_array_      = nullptr;
    T **             host_c_array_      = nullptr;
    cudaStream_t     copy_stream_;
    cudaEvent_t      copy_event_;

protected:
    // To prevent compiler warnings.
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    using nvinfer1::IPluginV2DynamicExt::enqueue;
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;
}; // end of MatchMatrixTensor

class MatchMatrixTensorPluginCreator : public nvinfer1::IPluginCreator
{
public:
    MatchMatrixTensorPluginCreator()
    {
        fc_.nbFields = attr_.size();
        fc_.fields   = attr_.data();
    }
    ~MatchMatrixTensorPluginCreator() {}
    const char *getPluginName() const
    {
        return MATCH_MATRIX_TENSOR_NAME;
    }
    const char *getPluginVersion() const
    {
        return MATCH_MATRIX_TENSOR_VERSION;
    }
    const nvinfer1::PluginFieldCollection *getFieldNames()
    {
        return &fc_;
    }

    nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
    {
        DEBUG_FUNC();

        nvinfer1::Weights w;
        int               h = 0, dim_t = 0;

        for (int i = 0; i < fc->nbFields; ++i)
        {
            auto        field = fc->fields[i];
            std::string field_name(field.name);

            if (field_name.compare("w") == 0)
            {
                w.values = field.data;
                w.count  = field.length;
                w.type   = trt_field_type_to_dtype(field.type);
            }

            if (field_name.compare("h") == 0)
            {
                assert(field.type == nvinfer1::PluginFieldType::kINT32);
                h = *reinterpret_cast<const int *>(field.data);
            }

            if (field_name.compare("dim_t") == 0)
            {
                assert(field.type == nvinfer1::PluginFieldType::kINT32);
                dim_t = *reinterpret_cast<const int *>(field.data);
            }
        }

        if (w.type == nvinfer1::DataType::kHALF)
        {
            return new MatchMatrixTensor<half>(name, w, h, dim_t, true);
        }
        return new MatchMatrixTensor<float>(name, w, h, dim_t, true);
    }
    nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength)
    {
        size_t offset = 0;
        auto   dtype  = nvinfer1::DataType::kFLOAT;
        trt_deserialize_value(serialData, serialLength, offset, dtype);

        if (dtype == nvinfer1::DataType::kHALF)
        {
            return new MatchMatrixTensor<half>(name, serialData, serialLength);
        }
        return new MatchMatrixTensor<float>(name, serialData, serialLength);
    }
    void setPluginNamespace(const char *pluginNamespace)
    {
        namespace_ = pluginNamespace;
    }
    const char *getPluginNamespace() const
    {
        return namespace_.c_str();
    }

private:
    static nvinfer1::PluginFieldCollection    fc_;
    static std::vector<nvinfer1::PluginField> attr_;
    std::string                               namespace_;
};

nvinfer1::PluginFieldCollection    MatchMatrixTensorPluginCreator::fc_ {};
std::vector<nvinfer1::PluginField> MatchMatrixTensorPluginCreator::attr_;

REGISTER_TENSORRT_PLUGIN(MatchMatrixTensorPluginCreator);

#if BUILD_UTEST
int max_seq_len(const std::vector<int> &lod)
{
    int max = 0;
    for (int i = 0; i < lod.size() - 1; ++i)
    {
        auto tmp = lod[i + 1] - lod[i];
        if (tmp > max)
            max = tmp;
    }
    return max;
}

template<typename T>
thrust::host_vector<T> padd_input(const thrust::host_vector<T> &unpadd,
                                  int                           emb_size,
                                  const std::vector<int> &      lod)
{
    int padd_seq_len = max_seq_len(lod);
    int num_seq      = lod.size() - 1;

    thrust::host_vector<T> padd;
    padd.resize(num_seq * padd_seq_len * emb_size);
    memset(padd.data(), 0, num_seq * padd_seq_len * emb_size * sizeof(T));

    size_t src_offset = unpadd.size();
    size_t dst_offset = num_seq * padd_seq_len * emb_size;

    for (int i = 0; i < num_seq; ++i)
    {
        const auto *src  = unpadd.data() + lod[i] * emb_size;
        auto *      dst  = padd.data() + i * padd_seq_len * emb_size;
        size_t      size = (lod[i + 1] - lod[i]) * emb_size * sizeof(T);
        memcpy(dst, src, size);
    }

    return padd;
}

template<typename T>
void print_2d(const thrust::host_vector<T> &x, int h, int w)
{
    assert(x.size() == w * h);
    for (int j = 0; j < h; ++j)
    {
        for (int i = 0; i < w; ++i)
        {
            std::cout << static_cast<float>(x[j * w + i]) << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename T>
int test1()
{
    // prepare ins and outs tensor in gpu, including size and lod
    int                    ix = 5, iy = 4, h = 2, dim_t = 2, batch_size = 4;
    thrust::host_vector<T> x, w, y;
    x.resize(ix * h);
    w.resize(h * dim_t * h);
    y.resize(iy * h);

    std::vector<int> x_lod, y_lod;
    x_lod = {0, 2, 5};
    y_lod = {0, 3, 4};

    int len = ix * h;
    for (int i = 0; i < len; ++i)
    {
        x[i] = static_cast<T>(static_cast<float>(i));
    }

    for (int i = 0; i < w.size(); ++i)
    {
        w[i] = static_cast<T>(static_cast<float>(i));
    }

    int len = iy * h;
    for (int i = 0; i < len; ++i)
    {
        y[i] = static_cast<T>(static_cast<float>(i));
    }

    // padding
    auto x_padd    = padd_input(x, h, x_lod);
    auto y_padd    = padd_input(y, h, y_lod);
    auto x_max_len = max_seq_len(x_lod);
    auto y_max_len = max_seq_len(y_lod);
    int  num_seq   = x_lod.size() - 1;

    //print_2d(x, 1, ix*h);
    //print_2d(x_padd, 1, num_seq*x_max_len*h);

    size_t workspace_size = (num_seq * x_max_len) * (dim_t * h) * sizeof(T);

    // H2D copy
    thrust::device_vector<T> dev_x = x_padd;
    thrust::device_vector<T> dev_y = y_padd;

    size_t                   out_count = dim_t * num_seq * x_max_len * y_max_len;
    thrust::device_vector<T> dev_out {out_count};

    // build
    auto builder = nvinfer1::createInferBuilder(gLogger);
    auto network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    auto input_x = network->addInput("x", trt_dtype<T>(), nvinfer1::Dims {3, {-1, -1, -1}});
    auto input_y = network->addInput("y", trt_dtype<T>(), nvinfer1::Dims {3, {-1, -1, -1}});

    // add plugin
    nvinfer1::Weights trt_w;
    trt_w.type                              = trt_dtype<T>();
    trt_w.values                            = thrust::raw_pointer_cast(w.data());
    trt_w.count                             = w.size();
    auto                             plugin = new MatchMatrixTensor<T>("MatchMatrixTensor", trt_w, h, dim_t);
    std::vector<nvinfer1::ITensor *> inputs {input_x, input_y};
    auto                             custom_layer = network->addPluginV2(inputs.data(), inputs.size(), *plugin);

    network->markOutput(*(custom_layer->getOutput(0)));

    auto profile = builder->createOptimizationProfile();

    profile->setDimensions(input_x->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims {3, {1, 1, 1}});
    profile->setDimensions(input_x->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims {3, {num_seq, x_max_len, h}});
    profile->setDimensions(input_x->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims {3, {num_seq, x_max_len, h}});
    profile->setDimensions(input_y->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims {3, {1, 1, 1}});
    profile->setDimensions(input_y->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims {3, {num_seq, y_max_len, h}});
    profile->setDimensions(input_y->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims {3, {num_seq, y_max_len, h}});

    auto config = builder->createBuilderConfig();
    config->addOptimizationProfile(profile);
    config->setMaxWorkspaceSize(workspace_size);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto engine = builder->buildEngineWithConfig(*network, *config);

    delete plugin;
    config->destroy();
    network->destroy();
    builder->destroy();

    // infer
    auto context = engine->createExecutionContext();
    context->setBindingDimensions(0, nvinfer1::Dims {3, {num_seq, x_max_len, h}});
    context->setBindingDimensions(1, nvinfer1::Dims {3, {num_seq, y_max_len, h}});

    assert(context->allInputDimensionsSpecified());
    std::vector<void *> bindings = {
        thrust::raw_pointer_cast(dev_x.data()),
        thrust::raw_pointer_cast(dev_y.data()),
        thrust::raw_pointer_cast(dev_out.data()),
    };
    auto status = context->executeV2(bindings.data());
    assert(status);

    // D2H copy
    thrust::host_vector<T> out = dev_out;

    std::vector<float> ref_results = {
        5,
        23,
        41,
        17,
        75,
        133,
        0,
        0,
        0,
    };

    //int len = out.size();
    //print_2d(out, 1, len);

    int len = out.size();
    for (int i = 0; i < len; ++i)
    {
        float out_fp32 = static_cast<float>(out[i]);
        assert(fabs(out_fp32 - ref_results[i]) < 1e-5);
    }

    context->destroy();

    auto serialized = engine->serialize();
    engine->destroy();

    std::string  fname = (trt_dtype<T>() == nvinfer1::DataType::kFLOAT) ? "engine.fp32.bin" : "engine.fp16.bin";
    std::fstream fs(fname, std::fstream::out | std::fstream::binary);
    fs.write(reinterpret_cast<const char *>(serialized->data()), serialized->size());
    fs.close();
    serialized->destroy();

    std::cout << "PASSED" << std::endl;
    return 0;
}

template<typename T>
int test2()
{
    // prepare ins and outs tensor in gpu, including size and lod
    int                    ix = 5, iy = 4, h = 2, dim_t = 2;
    thrust::host_vector<T> x, w, y;
    x.resize(ix * h);
    w.resize(h * dim_t * h);
    y.resize(iy * h);

    std::vector<int> x_lod, y_lod;
    x_lod = {0, 2, 5};
    y_lod = {0, 3, 4};

    int len = ix * h;
    for (int i = 0; i < len; ++i)
    {
        x[i] = static_cast<T>(static_cast<float>(i));
    }

    for (int i = 0; i < w.size(); ++i)
    {
        w[i] = static_cast<T>(static_cast<float>(i));
    }

    int len = iy * h;
    for (int i = 0; i < len; ++i)
    {
        y[i] = static_cast<T>(static_cast<float>(i));
    }

    // padding
    auto x_padd    = padd_input(x, h, x_lod);
    auto y_padd    = padd_input(y, h, y_lod);
    auto x_max_len = max_seq_len(x_lod);
    auto y_max_len = max_seq_len(y_lod);
    int  num_seq   = x_lod.size() - 1;

    // H2D copy
    thrust::device_vector<T> dev_x = x_padd;
    thrust::device_vector<T> dev_y = y_padd;

    size_t                   out_count = dim_t * num_seq * x_max_len * y_max_len;
    thrust::device_vector<T> dev_out {out_count};

    // runtime
    auto runtime = nvinfer1::createInferRuntime(gLogger);

    std::string  fname = (trt_dtype<T>() == nvinfer1::DataType::kFLOAT) ? "engine.fp32.bin" : "engine.fp16.bin";
    std::fstream fs(fname, std::fstream::in | std::fstream::binary);
    fs.seekg(0, fs.end);
    size_t fsize = fs.tellg();
    fs.seekg(0, fs.beg);
    char *serialized = new char[fsize];
    fs.read(serialized, fsize);
    fs.close();

    auto engine = runtime->deserializeCudaEngine(serialized, fsize);
    runtime->destroy();
    delete[] serialized;

    // infer
    auto context = engine->createExecutionContext();
    context->setBindingDimensions(0, nvinfer1::Dims {3, {num_seq, x_max_len, h}});
    context->setBindingDimensions(1, nvinfer1::Dims {3, {num_seq, y_max_len, h}});

    assert(context->allInputDimensionsSpecified());
    std::vector<void *> bindings = {
        thrust::raw_pointer_cast(dev_x.data()),
        thrust::raw_pointer_cast(dev_y.data()),
        thrust::raw_pointer_cast(dev_out.data()),
    };
    auto status = context->executeV2(bindings.data());
    assert(status);

    // D2H copy
    thrust::host_vector<T> out = dev_out;

    std::vector<float> ref_results = {
        5,
        23,
        41,
        17,
        75,
        133,
        0,
        0,
        0,
    };

    int len = out.size();
    for (int i = 0; i < len; ++i)
    {
        float out_fp32 = static_cast<float>(out[i]);
        assert(fabs(out_fp32 - ref_results[i]) < 1e-5);
    }

    context->destroy();
    engine->destroy();

    std::cout << "PASSED" << std::endl;
    return 0;
}

int main()
{
    printf("[main] cublas version: %d\n", CUBLAS_VERSION);
    test1<float>();
    test1<half>();
    test2<float>();
    test2<half>();
    return 0;
}
#endif
