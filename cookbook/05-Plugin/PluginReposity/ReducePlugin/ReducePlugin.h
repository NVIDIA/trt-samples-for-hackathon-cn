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
#include <cuda_fp16.h>

#define CEIL(x, y) (((x) + (y)-1) / (y))
#define MIN_FLOAT  (-1024.0f)

using namespace nvinfer1;

class ReducePlugin : public IPluginV2Ext
{
private:
    struct
    {
        int nRow;
        int nReduce;
        int nCol;
        int isFp16;
        int isSum;
    } m;

public:
    ReducePlugin(int isSum)
    {
        m.isSum = isSum;
    }

    int getNbOutputs() const override
    {
        return 1;
    }

    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const
    {
        return inputTypes[0];
    }

    Dims getOutputDimensions(int index, const Dims *pInputDim, int nInputDim) override
    {
        int  nDim = pInputDim[0].nbDims;
        Dims dd   = Dims {nDim - 1, {0}};
        for (int i = 0; i < nDim - 1; ++i)
            dd.d[i] = pInputDim[0].d[i];
        dd.d[nDim - 2] = pInputDim[0].d[nDim - 1];
        return dd;
    }

    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        return (type == DataType::kHALF || type == DataType::kFLOAT) && format == PluginFormat::kNCHW;
    }

    size_t getWorkspaceSize(int nBatch) const override
    {
        return 0;
    }

    void configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast, const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
    {
        int nDim = inputDims[0].nbDims, nRow = 1;
        for (int i = 0; i < nDim - 2; ++i)
            nRow *= inputDims[0].d[i];
        m.nRow    = nRow;
        m.nReduce = inputDims[0].d[nDim - 2];
        m.nCol    = inputDims[0].d[nDim - 1];
        m.isFp16  = (inputTypes[0] == DataType::kHALF);
    }

    int enqueue(int nBatch, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;
    ReducePlugin(const void *buffer, size_t length)
    {
        memcpy(&m, buffer, sizeof(m));
    }
    IPluginV2Ext *clone() const override
    {
        return new ReducePlugin(&m, sizeof(m));
    }
    int initialize() override
    {
        return 0;
    }
    void terminate() override {}
    void destroy() override
    {
        delete this;
    }
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }
    bool canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }
    void        attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) {}
    void        detachFromContext() {}
    const char *getPluginType() const override
    {
        return "ReducePlugin";
    }
    const char *getPluginVersion() const override
    {
        return "0";
    }
    void        setPluginNamespace(const char *szNamespace) override {}
    const char *getPluginNamespace() const override
    {
        return "";
    }
    virtual size_t getSerializationSize() const override
    {
        return sizeof(m);
    }
    virtual void serialize(void *buffer) const override
    {
        memcpy(buffer, &m, sizeof(m));
    }
};

class ReducePluginCreator : public IPluginCreator
{
public:
    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override
    {
        int isSum = 0;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            if (!strcmp(fc->fields[i].name, "isSum"))
                isSum = *(int *)fc->fields[i].data;
        }
        return new ReducePlugin(isSum);
    }

    const char *getPluginName() const override
    {
        return "ReducePlugin";
    }
    const char *getPluginVersion() const override
    {
        return "0";
    }
    void        setPluginNamespace(const char *szNamespace) override {}
    const char *getPluginNamespace() const override
    {
        return "";
    }
    const PluginFieldCollection *getFieldNames() override
    {
        return nullptr;
    }
    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override
    {
        return new ReducePlugin(serialData, serialLength);
    }
};
