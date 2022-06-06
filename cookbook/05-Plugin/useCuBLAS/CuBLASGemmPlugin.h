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
#include <cublas_v2.h>
#include <iostream>
#include <string>
#include <vector>

#ifdef DEBUG
    #define WHERE_AM_I()                               \
        do                                             \
        {                                              \
            printf("[%s]: this=%p\n", __func__, this); \
        } while (0);
#else
    #define WHERE_AM_I()
#endif // ifdef DEBUG

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

namespace
{
static const char *PLUGIN_NAME {"CuBLASGemm"};
static const char *PLUGIN_VERSION {"1"};
} // namespace

namespace nvinfer1
{
class CuBLASGemmPlugin : public IPluginV2DynamicExt
{
private:
    const std::string name_;
    std::string       namespace_;

public:
    bool           bOwnWeight_ {false};
    int            nK_ {0}; // shape of the weight, B_{nK,nN}
    int            nN_ {0};
    float *        pGPUWeight_ {nullptr};
    Weights        weight_ {DataType::kFLOAT, nullptr, 0};
    cublasHandle_t handle_ {nullptr};

public:
    CuBLASGemmPlugin() = delete;
    CuBLASGemmPlugin(const std::string &name, Weights weight, int k, int n, bool needDeepCopy);
    CuBLASGemmPlugin(const std::string &name, const void *buffer, size_t length);
    ~CuBLASGemmPlugin();

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

    //Method inherited from IPluginV2DynamicExt
    IPluginV2DynamicExt *clone() const noexcept override;
    DimsExprs            getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;
    bool                 supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void                 configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override;
    size_t               getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override;
    int32_t              enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
};

class CuBLASGemmPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    CuBLASGemmPluginCreator();
    ~CuBLASGemmPluginCreator();
    const char *                 getPluginName() const noexcept override;
    const char *                 getPluginVersion() const noexcept override;
    const PluginFieldCollection *getFieldNames() noexcept override;
    IPluginV2 *                  createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;
    IPluginV2 *                  deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override;
    void                         setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *                 getPluginNamespace() const noexcept override;
};

} // namespace nvinfer1
