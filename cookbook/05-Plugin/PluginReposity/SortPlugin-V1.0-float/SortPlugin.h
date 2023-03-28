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

#include "cub/device/device_radix_sort.cuh"

#include <NvInfer.h>

using namespace cub;
using namespace nvinfer1;

class SortPlugin : public IPluginV2
{
private:
    class inter
    {
    public:
        inter() {}
        ~inter() {}
        DataType dataType;
        int      nElement;
        int      width;
        size_t   tempSpaceSize;
        size_t   tempSpaceSizeAlign;
        void *   tempSpace;
        int *    value;
        bool     descending;
    };
    class inter m;

public:
    SortPlugin(int descending)
    {
        m.descending = (bool)descending;
    }

    SortPlugin(const void *buffer, size_t length)
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

    IPluginV2 *clone() const override
    {
        return new SortPlugin(&m, sizeof(m));
    }

    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
    }

    int getNbOutputs() const override
    {
        return 2;
    }

    Dims getOutputDimensions(int index, const Dims *pInputDim, int nInputDim) override
    {
        return pInputDim[index];
    }

    void configureWithFormat(const Dims *pInputDim, int nInputDim, const Dims *pOutputDim, int nOutputDim, DataType dataType, PluginFormat pluginFormat, int maxBatchSize) override
    {
        m.dataType      = dataType; // not used
        m.nElement      = pInputDim[1].d[0];
        m.width         = pInputDim[1].d[1];
        m.tempSpaceSize = 0;
        m.tempSpace     = nullptr;

        DoubleBuffer<float> dKey;
        DoubleBuffer<float> dValue;
        DeviceRadixSort::SortPairs(m.tempSpace, m.tempSpaceSize, dKey, dValue, m.nElement);
        m.tempSpaceSizeAlign = (m.tempSpaceSize + 1023) / 1024 * 1024;
        printf("+--------nElement = %d, width = %d, workSpaceSize = %ld\n", m.nElement, m.width, m.tempSpaceSizeAlign);
    }

    size_t getWorkspaceSize(int nBatch) const override
    {
        return m.tempSpaceSizeAlign;
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
        return "SortPlugin";
    }

    const char *getPluginVersion() const override
    {
        return "0";
    }
};

class SortPluginCreator : public IPluginCreator
{
public:
    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override
    {
        return new SortPlugin(serialData, serialLength);
    }

    const char *getPluginName() const override
    {
        return "SortPlugin";
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

    const PluginFieldCollection *getFieldNames() override
    {
        return nullptr;
    }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override
    {
        int descending = false;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            if (!strcmp(fc->fields[i].name, "descending"))
                descending = *(int *)fc->fields[i].data;
        }
        return new SortPlugin(descending);
    }
};
