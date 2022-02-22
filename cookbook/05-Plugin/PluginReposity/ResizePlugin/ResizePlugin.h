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

#define ALIGNSIZE  1024
#define ALIGNED(x) (((x) + ALIGNSIZE - 1) / ALIGNSIZE * ALIGNSIZE)

using namespace nvinfer1;

class ResizePlugin : public IPluginV2
{
private:
    struct
    {
        DataType dataType; // not used
        int      h1;
        int      w1;
        int      h2;
        int      w2;
        int      c;
    } m;

public:
    ResizePlugin(int hOut, int wOut)
    {
        m.h2 = hOut;
        m.w2 = wOut;
    }

    ResizePlugin(const void *buffer, size_t length)
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
        return new ResizePlugin(&m, sizeof(m));
    }

    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
    }

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims *pInputDim, int nInputDim) override
    {
        return Dims {3, {pInputDim[0].d[0], m.h2, m.w2}};
    }

    void configureWithFormat(const Dims *pInputDim, int nInputDim, const Dims *pOutputDim, int nOutputDim, DataType dataType, PluginFormat pluginFormat, int maxBatchSize) override
    {
        m.dataType = dataType;
        m.c        = pInputDim[0].d[0];
        m.h1       = pInputDim[0].d[1];
        m.w1       = pInputDim[0].d[2];
    }

    size_t getWorkspaceSize(int nBatch) const override
    {
        return 0;
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
        return "ResizePlugin";
    }

    const char *getPluginVersion() const override
    {
        return "0";
    }
};

class ResizePluginCreator : public IPluginCreator
{
public:
    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override
    {
        return new ResizePlugin(serialData, serialLength);
    }

    const char *getPluginName() const override
    {
        return "ResizePlugin";
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
        int hOut = 0, wOut = 0; // the size of output tensor
        for (int i = 0; i < fc->nbFields; i++)
        {
            if (!strcmp(fc->fields[i].name, "hOut"))
                hOut = *(int *)fc->fields[i].data;
            if (!strcmp(fc->fields[i].name, "wOut"))
                wOut = *(int *)fc->fields[i].data;
        }
        return new ResizePlugin(hOut, wOut);
    }
};
