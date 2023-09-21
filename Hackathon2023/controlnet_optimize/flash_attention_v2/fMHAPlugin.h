/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

#include <nvtx3/nvToolsExt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <cassert>
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <iostream>
#include <memory>

#include <fmha_api.h>

// +------- Debug wrapper --------------------------------------------------------------------------
#if false
#define WHERE_AM_I() do {printf("[%s]: this=->%p\n",__func__,this);} while(0);
#else
#define WHERE_AM_I()
#endif // DEBUG

#define CHECK(call) check(call, __LINE__, __FILE__)

inline bool check(cudaError_t e, int iLine, const char *szFile)
{
    if (e != cudaSuccess)
    {
        std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << std::endl;
        return false;
    }
    return true;
}

template <typename T>
struct CudaDeleter
{
    void operator()(T* buf)
    {
        cudaFree(buf);
    }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter<T>>;

template <typename T>
using cuda_shared_ptr = std::shared_ptr<T>;

template <typename T>
void make_cuda_shared(cuda_shared_ptr<T>& ptr, void* cudaMem)
{
    ptr.reset(static_cast<T*>(cudaMem), CudaDeleter<T>());
}

namespace
{
static const char *PLUGIN_NAME {"fMHAPlugin"};
static const char *PLUGIN_VERSION {"1"};
} // namespace

namespace nvinfer1
{
class fMHAPlugin : public IPluginV2DynamicExt
{
private:
    struct {
        float dropout_p;
        float scale;
        int causal;
        int return_attn_probs;
        int32_t mOptBatchSize{};
        int32_t mOptSeqLenQ{};
        int32_t mOptSeqLenKV{};
        int32_t mMaxBatchSize{};
        DataType mDataType{DataType::kFLOAT};
    } m_;

    cuda_shared_ptr<void> mCuSeqLensQ;
    cuda_shared_ptr<void> mCuSeqLensKV;
    const std::string name_;
    std::string       namespace_;

public:
    fMHAPlugin() = delete;
    fMHAPlugin(const std::string &name, float dropout_p, float scale, int causal, int return_attn_probs);
    fMHAPlugin(const std::string &name, const void *buffer, size_t length);
    ~fMHAPlugin();

    void init();

    // Method inherited from IPluginV2
    const char *getPluginType() const noexcept override;
    const char *getPluginVersion() const noexcept override;
    int32_t     getNbOutputs() const noexcept override;
    int32_t     initialize() noexcept override;
    void        terminate() noexcept override;
    size_t      getSerializationSize() const noexcept override;
    void        serialize(void *buffer) const noexcept override;
    void        destroy() noexcept override;
    void        setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *getPluginNamespace() const noexcept override;

    // Method inherited from IPluginV2Ext
    DataType getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept override;
    void     attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept override;
    void     detachFromContext() noexcept override;

    // Method inherited from IPluginV2DynamicExt
    IPluginV2DynamicExt *clone() const noexcept override;
    DimsExprs            getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;
    bool                 supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void                 configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override;
    size_t               getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override;
    int32_t              enqueue(const PluginTensorDesc *inputDesc,
                                 const PluginTensorDesc *outputDesc,
                                 const void *const *     inputs,
                                 void *const *           outputs,
                                 void *                  workspace,
                                 cudaStream_t            stream) noexcept override;

protected:
    void allocateSeqlens(int32_t maxBatchSize);
    int32_t initializeSeqlens(int32_t b, int32_t s, void* cuSeqlensDev, cudaStream_t stream = 0);
};


class fMHAPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    fMHAPluginCreator();
    ~fMHAPluginCreator();
    const char *                 getPluginName() const noexcept override;
    const char *                 getPluginVersion() const noexcept override;
    const PluginFieldCollection *getFieldNames() noexcept override;
    IPluginV2 *                  createPlugin(const char *                 name,
                                              const PluginFieldCollection *fc) noexcept override;
    IPluginV2 *                  deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override;
    void                         setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *                 getPluginNamespace() const noexcept override;
};

} // namespace nvinfer1
