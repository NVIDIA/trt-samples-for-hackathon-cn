#pragma once

#include "NvInfer.h"

#include <assert.h>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>

using namespace nvinfer1;
//using namespace std;

class UpsamplePlugin : public IPluginV2
{
public:
    UpsamplePlugin(int nScaleFactor, bool bNearest);

    virtual ~UpsamplePlugin() {}

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims *pInputDim, int nInputDim) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    void configureWithFormat(const Dims *pInputDim, int nInputDim, const Dims *pOutputDim, int nOutputDim, DataType dataType, PluginFormat pluginFormat, int maxBatchSize) override;

    int initialize() override
    {
        return 0;
    }

    virtual void terminate() override {}

    virtual size_t getWorkspaceSize(int nBatch) const override
    {
        return 0;
    }

    virtual int enqueue(int nBatch, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;

    virtual size_t getSerializationSize() const override
    {
        return 0;
    }

    virtual void serialize(void *buffer) const override {}

    const char *getPluginType() const override
    {
        return "Upsample_Plugin";
    }

    const char *getPluginVersion() const override
    {
        return "0";
    }

    void destroy()
    {
        delete this;
    }

    IPluginV2 *clone() const override
    {
        return new UpsamplePlugin(nScaleFactor, bNearest);
    }

    virtual void setPluginNamespace(const char *pluginNamespace) {}

    virtual const char *getPluginNamespace() const
    {
        return nullptr;
    }

private:
    int  nChannel = 0, nSrcWidth = 0, nSrcHeight = 0;
    int  nScaleFactor;
    bool bNearest;
};
