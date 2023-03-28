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

#include "cookbookHelper.cuh"

namespace
{
static const char *PLUGIN_NAME {"AddScalar"};
static const char *PLUGIN_VERSION {"1"};
} // namespace

namespace nvinfer1
{
class AddScalarPlugin : public IPluginV2Ext // deprecated since TensorRT 8.5
{
private:
    const std::string name_;
    std::string       namespace_;
    struct
    {
        float scalar;
        int   nElement;
    } m_;

public:
    AddScalarPlugin() = delete;
    AddScalarPlugin(const std::string &name, float scalar);
    AddScalarPlugin(const std::string &name, const void *buffer, size_t length);
    ~AddScalarPlugin();

    // Method inherited from IPluginV2
    const char *getPluginType() const noexcept override;
    const char *getPluginVersion() const noexcept override;
    int32_t     getNbOutputs() const noexcept override;
    Dims        getOutputDimensions(int32_t index, Dims const *inputs, int32_t nbInputDims) noexcept override;
    bool        supportsFormat(DataType type, PluginFormat format) const noexcept override;
    int32_t     initialize() noexcept override;
    void        terminate() noexcept override;
    size_t      getWorkspaceSize(int32_t maxBatchSize) const noexcept override;
    //int32_t enqueue(int32_t batchSize, const void * const * inputs, void ** outputs, void * workspace, cudaStream_t stream) noexcept override;        // TensorRT 7
    int32_t     enqueue(int32_t batchSize, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override; // TensorRT 8
    size_t      getSerializationSize() const noexcept override;
    void        serialize(void *buffer) const noexcept override;
    void        destroy() noexcept override;
    void        setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *getPluginNamespace() const noexcept override;

    // Method inherited from IPluginV2Ext
    DataType      getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept override;
    bool          isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool *inputIsBroadcasted, int32_t nbInputs) const noexcept override;
    bool          canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;
    void          configurePlugin(Dims const *inputDims, int32_t nbInputs, Dims const *outputDims, int32_t nbOutputs, DataType const *inputTypes, DataType const *outputTypes, bool const *inputIsBroadcast, bool const *outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept;
    void          attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept override;
    void          detachFromContext() noexcept override;
    IPluginV2Ext *clone() const noexcept override;

protected:
    // To prevent compiler warnings
    using nvinfer1::IPluginV2::enqueue;
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
};

class AddScalarPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    AddScalarPluginCreator();
    ~AddScalarPluginCreator();
    const char *                 getPluginName() const noexcept override;
    const char *                 getPluginVersion() const noexcept override;
    const PluginFieldCollection *getFieldNames() noexcept override;
    IPluginV2Ext *               createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;
    IPluginV2Ext *               deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override;
    void                         setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *                 getPluginNamespace() const noexcept override;
};

} // namespace nvinfer1
