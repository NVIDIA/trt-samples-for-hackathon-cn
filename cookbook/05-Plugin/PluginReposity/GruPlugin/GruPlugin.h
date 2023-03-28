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

#ifndef _GRU_PLUGIN_H_
#define _GRU_PLUGIN_H_

#include "cublas_v2.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <iostream>
#include <string>

#define CUDA_MEM_ALIGN 256

#define CEIL(X, Y) (((X) + (Y)-1) / (Y) * (Y))

#define CHECK_CUDA_ERROR(cond)                                                                             \
    do                                                                                                     \
    {                                                                                                      \
        if (cond != cudaSuccess)                                                                           \
        {                                                                                                  \
            std::cout << "CUDA Error: " << cudaGetErrorString(cond) << " line: " << __LINE__ << std::endl; \
        }                                                                                                  \
    } while (0)

#define CHECK_CUBLAS_ERROR(cond)                                                                            \
    do                                                                                                      \
    {                                                                                                       \
        if (cond != CUBLAS_STATUS_SUCCESS)                                                                  \
        {                                                                                                   \
            std::cout << "CUBLAS Error: " << _cudaGetErrorEnum(cond) << " line: " << __LINE__ << std::endl; \
        }                                                                                                   \
    } while (0)

#ifdef CUBLAS_API_H_
// cublas API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
        return "<unknown>";
    }
}
#endif

using namespace nvinfer1;

class GruPlugin : public IPluginV2DynamicExt
{
public:
    GruPlugin(const std::string name, const int inputSize, const int hiddenSize, float *x_weights, float *h_weights, float *bias);
    GruPlugin(const std::string name, const void *data, size_t length);
    GruPlugin(const GruPlugin &obj);
    GruPlugin() = delete;
    ~GruPlugin() override;

    const char *getPluginType() const override;
    const char *getPluginVersion() const override;
    int32_t     getNbOutputs() const override;
    int32_t     initialize() override;
    void        terminate() override;
    size_t      getSerializationSize() const override;
    void        serialize(void *buffer) const override;
    void        destroy() override;
    void        setPluginNamespace(const char *pluginNamespace) override;
    const char *getPluginNamespace() const override;

    DataType getOutputDataType(int32_t index, const DataType *inputTypes, int32_t nbInputs) const override;
    void     attachToContext(cudnnContext *, cublasContext *, IGpuAllocator *) override;
    void     detachFromContext() override;

    IPluginV2DynamicExt *clone() const override;
    DimsExprs            getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) override;
    bool                 supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) override;
    void                 configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) override;
    size_t               getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const override;
    int32_t              enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;

private:
    int            mInputSize;
    int            mHiddenSize;
    float *        mWeightsX_h      = nullptr; // x weights(host memory)
    float *        mWeightsX_d      = nullptr; // x weights(device memory)
    __half *       mWeightsX_half_d = nullptr; // x weights in fp16(device memory)
    float *        mWeightsH_h      = nullptr; // h weights(host memory)
    float *        mWeightsH_d      = nullptr; // h weights(device memory)
    __half *       mWeightsH_half_d = nullptr; // h weights in fp16(device memory)
    float *        mBias_h          = nullptr; // bias(host memory)
    float *        mBias_d          = nullptr; // bias(device memory)
    __half *       mBias_half_d     = nullptr; // bias in half(device memory)
    cublasHandle_t mCuBlasHandle    = nullptr;
    std::string    mLayerName;
    std::string    mPluginNamespace;
};

class GruPluginCreator : public IPluginCreator
{
public:
    GruPluginCreator();
    ~GruPluginCreator() override = default;
    const char *                 getPluginName() const override;
    const char *                 getPluginVersion() const override;
    const PluginFieldCollection *getFieldNames() override;
    IPluginV2 *                  createPlugin(const char *name, const PluginFieldCollection *fc) override;
    IPluginV2 *                  deserializePlugin(const char *name, const void *serialData, size_t serialLength) override;
    void                         setPluginNamespace(const char *pluginNamespace) override;
    const char *                 getPluginNamespace() const override;

private:
    std::string mPluginNamespace;
};

#endif // _GRU_PLUGIN_H_
