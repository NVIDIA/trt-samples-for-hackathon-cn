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
#include <cstdio>

#define ALIGNSIZE  1024
#define ALIGNED(x) (((x) + ALIGNSIZE - 1) / ALIGNSIZE * ALIGNSIZE)

using namespace nvinfer1;

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

class CCLPlugin : public IPluginV2DynamicExt
{
private:
    class inter
    {
    public:
        inter() {}
        ~inter() {}
        int   height;
        int   width;
        float minPixelScore;
        float minLinkScore;
        int   minArea;
        int   maxCount; // not used
    };
    class inter m;

public:
    CCLPlugin(float minPixelScore, float minLinkScore, int minArea, int maxCount)
    {
        m.minPixelScore = minPixelScore;
        m.minLinkScore  = minLinkScore;
        m.minArea       = minArea;
        m.maxCount      = maxCount;
    }

    CCLPlugin(const void *buffer, size_t length)
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

    IPluginV2DynamicExt *clone() const override
    {
        return new CCLPlugin(&m, sizeof(m));
    }

    int getNbOutputs() const override
    {
        return 2;
    }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder) override
    {
        if (outputIndex == 0)
            return inputs[0];

        DimsExprs out;
        out.nbDims = 1;
        out.d[0]   = inputs[0].d[0];
        return out;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) override
    {
        if (inOut[pos].format != TensorFormat::kLINEAR)
            return false;
        if (pos < 2)
            return inOut[pos].type == DataType::kFLOAT;
        return inOut[pos].type == DataType::kINT32;
    }

    DataType getOutputDataType(int outputIndex, const DataType *inputTypes, int nbInputs) const override
    {
        return DataType::kINT32;
    }

    void configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs) override
    {
        m.height = in[0].desc.dims.d[0];
        m.width  = in[0].desc.dims.d[1];
    }

    size_t getWorkspaceSize(const PluginTensorDesc *input, int nbInput, const PluginTensorDesc *output, int nbOutput) const override
    {
        return ALIGNED(sizeof(int) * input[0].dims.d[0] * (m.height + 2) * (m.width + 2)) +
               ALIGNED(sizeof(char8) * input[0].dims.d[0] * (m.height + 2) * (m.width + 2)) +
               ALIGNED(sizeof(int) * input[0].dims.d[0]) +
               ALIGNED(sizeof(int) * input[0].dims.d[0]);
    }

    int enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;

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

class CCLPluginCreator : public IPluginCreator
{
public:
    IPluginV2DynamicExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override
    {
        return new CCLPlugin(serialData, serialLength);
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

    const PluginFieldCollection *getFieldNames() override
    {
        return nullptr;
    }

    IPluginV2DynamicExt *createPlugin(const char *name, const PluginFieldCollection *fc) override
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
        return new CCLPlugin(minPixelScore, minLinkScore, minArea, maxCount);
    }
};
