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

#include "device_launch_parameters.h"

#include <NvInfer.h>
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdint.h>

#define ALIGNSIZE  1024
#define ALIGNED(x) (((x) + ALIGNSIZE - 1) / ALIGNSIZE * ALIGNSIZE)

union char8
{
    int2 i2;
    char c8[8];
};

struct int8
{
    int             value[8];
    __device__ int &operator[](int i)
    {
        return value[i];
    }
};

class LayerNormPlugin : public nvinfer1::IPluginV2
{
private:
    class inter
    {
    public:
        inter() {}
        ~inter() {}
        nvinfer1::DataType dataType; // not used
        int                height;
        int                width;
        float              minPixelScore;
        float              minLinkScore;
        int                minArea;
        int                maxCount; // not used
    };
    class inter m;

public:
    LayerNormPlugin(float minPixelScore, float minLinkScore, int minArea, int maxCount)
    {
        m.minPixelScore = minPixelScore;
        m.minLinkScore  = minLinkScore;
        m.minArea       = minArea;
        m.maxCount      = maxCount;
    }

    LayerNormPlugin(const void *buffer, size_t length)
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

    nvinfer1::IPluginV2 *clone() const override
    {
        return new LayerNormPlugin(&m, sizeof(m));
    }

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override
    {
        return type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kNCHW;
    }

    int getNbOutputs() const override
    {
        return 2;
    }

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *pInputDim, int nInputDim) override
    {
        return (index == 0) ? pInputDim[0] : nvinfer1::Dims {1, {1}};
    }

    void configureWithFormat(const nvinfer1::Dims *pInputDim, int nInputDim, const nvinfer1::Dims *pOutputDim, int nOutputDim, nvinfer1::DataType dataType, nvinfer1::PluginFormat pluginFormat, int maxBatchSize) override
    {
        m.dataType = dataType;
        m.height   = pInputDim[0].d[0];
        m.width    = pInputDim[0].d[1];
    }

    size_t getWorkspaceSize(int nBatch) const override
    {
        return ALIGNED(sizeof(int) * nBatch * (m.height + 2) * (m.width + 2)) +
               ALIGNED(sizeof(char8) * nBatch * (m.height + 2) * (m.width + 2)) +
               ALIGNED(sizeof(int) * nBatch) +
               ALIGNED(sizeof(int) * nBatch);
    }

    int enqueue(int nBatch, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;

    int initialize() override
    {
        return 0;
    }

    void terminate() override {}

    void destroy() override
    {
        delete this;
    }

    void setPluginNamespace(const char *szNamespace) override {}

    const char *getPluginNamespace() const override
    {
        return "";
    }

    const char *getPluginType() const override
    {
        return "CCLPlugin";
    }

    const char *getPluginVersion() const override
    {
        return "0";
    }
};

class AddPluginCreator : public nvinfer1::IPluginCreator
{
public:
    nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override
    {
        return new LayerNormPlugin(serialData, serialLength);
    }

    const char *getPluginName() const override
    {
        return "CCLPlugin";
    }

    const char *getPluginVersion() const override
    {
        return "0";
    }

    void setPluginNamespace(const char *szNamespace) override {}

    const char *getPluginNamespace() const override
    {
        return "";
    }

    const nvinfer1::PluginFieldCollection *getFieldNames() override
    {
        return nullptr;
    }

    nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) override
    {
        float minPixelScore = 1, minLinkScore = 1;
        int   minArea = 0, maxCount = 65536;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            if (!strcmp(fc->fields[i].name, "minPixelScore"))
                minPixelScore = *(float *)fc->fields[i].data;
            if (!strcmp(fc->fields[i].name, "minLinkScore"))
                minLinkScore = *(float *)fc->fields[i].data;
            if (!strcmp(fc->fields[i].name, "minArea"))
                minArea = *(int *)fc->fields[i].data;
            if (!strcmp(fc->fields[i].name, "maxCount"))
                maxCount = *(int *)fc->fields[i].data;
        }
        return new LayerNormPlugin(minPixelScore, minLinkScore, minArea, maxCount);
    }
};
