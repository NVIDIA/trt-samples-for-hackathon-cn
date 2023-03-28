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

#include "MultinomialDistributionPlugin.h"

// 用于计算的 kernel
template<int n>
__global__ void sampleSmallKernel(float *pDeviceProbabilityColumn, float *pTargetRandomValue, int *pDeviceIndex, float *pDeviceEntropy)
{
    int bx = blockIdx.x, tx = threadIdx.x;
    if (tx >= n)
    {
        return;
    }

    typedef cub::WarpScan<float, n>           WarpScan; // 由概率分布列计算经验分布函数
    __shared__ typename WarpScan::TempStorage tempScan;
    __shared__ float                          probList[n];
    probList[tx]     = pDeviceProbabilityColumn[bx * n + tx];
    float &tDataScan = probList[tx];
    WarpScan(tempScan).InclusiveSum(tDataScan, tDataScan);

    tDataScan /= probList[n - 1]; // 若输入分布列没有归一化，则在这里除以闭前缀和的最后一个元素，以归一化
    __syncthreads();
    //if(tx == 0)
    //    printf("(%4d,%2d,%5d)\t%f\t%f\n",bx,tx,id,tDataScan,probList[n-1]);

    float sample = pTargetRandomValue[bx]; // sample ~ U[0,1]

    typedef cub::WarpReduce<int>                WarpReduce; // 找到首个累计概率大于 sample 的下标，作为样本值
    __shared__ typename WarpReduce::TempStorage tempReduce;
    __shared__ int                              pCompareList[n];
    pCompareList[tx] = int(sample >= tDataScan);
    __syncthreads();
    int &tDataReduce = pCompareList[tx];
    int  index       = WarpReduce(tempReduce).Sum(tDataReduce);
    if (tx == 0)
    {
        pDeviceIndex[bx]   = index;
        pDeviceEntropy[bx] = -__logf((index == 0) ? probList[index] : (probList[index] - probList[index - 1]));
        //printf("(%4d,%2d,%5d)\t%f\t%d\t%f\t%f\n",bx,tx,bx*n+tx,sample,index,probList[index],
        //                                         -__logf( (index==0) ? probList[index]:(probList[index]-probList[index-1]) ) );
    }
    return;
}

template<int n>
__global__ void sampleLargeKernel(float *pDeviceProbabilityColumn, float *pTargetRandomValue, int *pDeviceIndex, float *pDeviceEntropy, int *pCompareList)
{
    const int bx = blockIdx.x, tx = threadIdx.x;
    if (tx >= n)
    {
        return;
    }

    typedef cub::BlockScan<float, n>           BlockScan;
    __shared__ typename BlockScan::TempStorage tempScan;
    float &                                    tDataScan = pDeviceProbabilityColumn[bx * n + tx];
    BlockScan(tempScan).InclusiveSum(tDataScan, tDataScan);
    __syncthreads(); // 必须同步

    pDeviceProbabilityColumn[bx * n + tx] /= pDeviceProbabilityColumn[bx * n + n - 1];
    __syncthreads();

    pCompareList[bx * n + tx] = int(pTargetRandomValue[bx] >= tDataScan);
    __syncthreads();

    typedef cub::BlockReduce<int, n>             BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempReduce;
    int &                                        tDataReduce = pCompareList[bx * n + tx];
    int                                          index       = min(BlockReduce(tempReduce).Sum(tDataReduce), n - 1);
    __syncthreads();

    if (tx == 0)
    {
        pDeviceIndex[bx]   = index;
        pDeviceEntropy[bx] = -__logf((index == 0) ? pDeviceProbabilityColumn[bx * n] : (pDeviceProbabilityColumn[bx * n + index] - pDeviceProbabilityColumn[bx * n + index - 1]));
        //printf("(%4d,%2d,%5d)\t%f\t%d\t%f\t%f\n",bx,tx,bx*n+tx,sample,index,probList[max(bx*n,bx*n+index-1], pDeviceEntropy[bx]);
    }
    return;
}

template<int n>
__global__ void sampleSmallKernel(__half *pDeviceProbabilityColumn, float *pTargetRandomValue, int *pDeviceIndex, __half *pDeviceEntropy)
{
    const int bx = blockIdx.x, tx = threadIdx.x;
    if (tx >= n)
    {
        return;
    }

    __shared__ __half probList[n]; // 一行一个分布列
    probList[tx] = pDeviceProbabilityColumn[bx * n + tx];
    typedef cub::WarpScan<__half, n>          WarpScan; // 由概率分布列计算经验分布函数
    __shared__ typename WarpScan::TempStorage tempScan;
    __half &                                  tDataScan = probList[tx];
    WarpScan(tempScan).InclusiveSum(tDataScan, tDataScan);

    tDataScan /= probList[n - 1]; // 若输入分布列没有归一化，则在这里除以闭前缀和的最后一个元素，以归一化
    //__syncthreads();
    //if(tx == 0)
    //printf("(%4d,%2d,%5d)\t%f\t%f\t%f\n",bx,tx,bx*n+tx, probList[0],probList[n/2], probList[n-1]);

    float sample = pTargetRandomValue[bx]; // sample ~ U[0,1]
    __syncthreads();

    __shared__ int pCompareList[n]; // 存放分布列一行的比较结果
    pCompareList[tx] = int(sample >= __half2float(tDataScan));
    typedef cub::WarpReduce<int>                WarpReduce; // 找到首个累计概率大于 sample 的分布函数的下标，作为输出样本
    __shared__ typename WarpReduce::TempStorage tempReduce;
    int &                                       tDataReduce = pCompareList[tx];
    int                                         index       = min(WarpReduce(tempReduce).Sum(tDataReduce), n - 1);

    if (tx == 0) // 保存样本和交叉熵值
    {
        pDeviceIndex[bx]   = index;
        pDeviceEntropy[bx] = __half2float(-hlog((index == 0) ? probList[0] : (probList[index] - probList[index - 1])));
        //printf("(%4d,%2d,%5d)\t%f\t%d\t%f\t%f\n",bx,tx,bx*n+tx, sample,index,probList[max(0,index-1)],pDeviceEntropy[bx]);
    }
    return;
}

template<int n>
__global__ void sampleLargeKernel(__half *pDeviceProbabilityColumn, float *pTargetRandomValue, int *pDeviceIndex, __half *pDeviceEntropy, int *pCompareList)
{
    const int bx = blockIdx.x, tx = threadIdx.x;
    if (tx >= n)
    {
        return;
    }

    typedef cub::BlockScan<__half, n>          BlockScan;
    __shared__ typename BlockScan::TempStorage tempScan;
    __half &                                   tDataScan = pDeviceProbabilityColumn[bx * n + tx];
    BlockScan(tempScan).InclusiveSum(tDataScan, tDataScan);
    __syncthreads(); // 必须同步

    pDeviceProbabilityColumn[bx * n + tx] /= pDeviceProbabilityColumn[bx * n + n - 1];
    __syncthreads();

    pCompareList[bx * n + tx] = int(pTargetRandomValue[bx] >= __half2float(tDataScan));
    __syncthreads();

    typedef cub::BlockReduce<int, n>             BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempReduce;
    int &                                        tDataReduce = pCompareList[bx * n + tx];
    int                                          index       = min(BlockReduce(tempReduce).Sum(tDataReduce), n - 1);
    __syncthreads();

    if (tx == 0)
    {
        pDeviceIndex[bx]   = int(index);
        pDeviceEntropy[bx] = __half2float(-hlog((index == 0) ? (pDeviceProbabilityColumn[bx * n]) : (pDeviceProbabilityColumn[bx * n + index] - pDeviceProbabilityColumn[bx * n + index - 1])));
        //printf("(%4d,%2d,%5d)\t%f\t%d\t%f\t%f\n",bx,tx,bx*n+tx,sample,index,probList[max(bx*n,bx*n+index-1], pDeviceEntropy[bx]);
    }
    return;
}

template __global__ void sampleSmallKernel<32>(float *, float *, int *, float *);
template __global__ void sampleLargeKernel<128>(float *, float *, int *, float *, int *);
template __global__ void sampleSmallKernel<32>(__half *, float *, int *, __half *);
template __global__ void sampleLargeKernel<128>(__half *, float *, int *, __half *, int *);

namespace nvinfer1
{
// class MultinomialDistributionPlugin
MultinomialDistributionPlugin::MultinomialDistributionPlugin(const std::string &name, int seed):
    name_(name)
{
    WHERE_AM_I();
    m_.seed         = seed;
    m_.firstEnqueue = 1;

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
    curandSetPseudoRandomGeneratorSeed(gen, m_.seed);
    m_.gen = gen;
}

MultinomialDistributionPlugin::MultinomialDistributionPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
    m_.firstEnqueue = 1;

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
    curandSetPseudoRandomGeneratorSeed(gen, m_.seed);
    m_.gen = gen;
}

MultinomialDistributionPlugin::~MultinomialDistributionPlugin()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *MultinomialDistributionPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new MultinomialDistributionPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int MultinomialDistributionPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 2;
}

DataType MultinomialDistributionPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return (index == 0) ? DataType::kINT32 : inputTypes[0];
}

DimsExprs MultinomialDistributionPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs ret = inputs[0];
    ret.nbDims    = 1;
    ret.d[1]      = exprBuilder.constant(1);
    return ret;
}

bool MultinomialDistributionPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    switch (pos)
    {
    case 0:
        return (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF) && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[1].type == DataType::kINT32 && inOut[1].format == TensorFormat::kLINEAR;
    case 2:
        return inOut[2].type == inOut[0].type && inOut[1].format == TensorFormat::kLINEAR;
    default: // should NOT be here!
        return false;
    }
    return false;
}

void MultinomialDistributionPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    m_.nMaxRow = in[0].max.d[0];
    m_.nMaxCol = in[0].max.d[1];
    return;
}

size_t MultinomialDistributionPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return ALIGN_TO(sizeof(float) * m_.nMaxRow, 1024) + ((m_.nMaxCol > 32) ? ALIGN_TO(sizeof(int) * m_.nMaxRow * m_.nMaxCol, 1024) : 0);
}

int32_t MultinomialDistributionPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    if (m_.firstEnqueue)
    {
        curandSetStream(m_.gen, stream);
        m_.firstEnqueue = 0;
    }

    int nRow = inputDesc[0].dims.d[0];
    int nCol = inputDesc[0].dims.d[1];

    curandGenerateUniform(m_.gen, (float *)workspace, nRow);

    if (inputDesc[0].type == DataType::kHALF)
    {
        switch (nCol)
        {
        case 4:
            (sampleSmallKernel<4>)<<<nRow, 32, 0, stream>>>((__half *)inputs[0], (float *)workspace, (int *)outputs[0], (__half *)outputs[1]);
            break;
        case 32:
            (sampleSmallKernel<32>)<<<nRow, 32, 0, stream>>>((__half *)inputs[0], (float *)workspace, (int *)outputs[0], (__half *)outputs[1]);
            break;
        case 128:
        {
            int *pCompareList = (int *)((char *)workspace + ALIGN_TO(sizeof(float) * m_.nMaxRow, 1024));
            (sampleLargeKernel<128>)<<<nRow, ALIGN_TO(nCol, 32), 0, stream>>>((__half *)inputs[0], (float *)workspace, (int *)outputs[0], (__half *)outputs[1], pCompareList);
            break;
        }
        default:
            printf("Failed matching nCol == %d in Fp16\n", nCol);
        }
    }
    else
    {
        switch (nCol)
        {
        case 4:
            (sampleSmallKernel<4>)<<<nRow, 32, 0, stream>>>((float *)inputs[0], (float *)workspace, (int *)outputs[0], (float *)outputs[1]);
            break;
        case 32:
            (sampleSmallKernel<32>)<<<nRow, 32, 0, stream>>>((float *)inputs[0], (float *)workspace, (int *)outputs[0], (float *)outputs[1]);
            break;
        case 128:
        {
            int *pCompareList = (int *)((char *)workspace + ALIGN_TO(sizeof(float) * m_.nMaxRow, 1024));
            (sampleLargeKernel<128>)<<<nRow, ALIGN_TO(nCol, 32), 0, stream>>>((float *)inputs[0], (float *)workspace, (int *)outputs[0], (float *)outputs[1], pCompareList);
            break;
        }
        default:
            printf("Failed matching nCol == %d in Fp32\n", nCol);
        }
    }
    return 0;
}

void MultinomialDistributionPlugin::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t MultinomialDistributionPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void MultinomialDistributionPlugin::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t MultinomialDistributionPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void MultinomialDistributionPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void MultinomialDistributionPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *MultinomialDistributionPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *MultinomialDistributionPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *MultinomialDistributionPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

void MultinomialDistributionPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void MultinomialDistributionPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class MultinomialDistributionPluginCreator
PluginFieldCollection    MultinomialDistributionPluginCreator::fc_ {};
std::vector<PluginField> MultinomialDistributionPluginCreator::attr_;

MultinomialDistributionPluginCreator::MultinomialDistributionPluginCreator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("seed", nullptr, PluginFieldType::kINT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

MultinomialDistributionPluginCreator::~MultinomialDistributionPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2 *MultinomialDistributionPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    int                          seed = 97;
    std::map<std::string, int *> parameterMap {{"seed", &seed}};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const float *>(fc->fields[i].data);
        }
    }
    return new MultinomialDistributionPlugin(name, seed);
}

IPluginV2 *MultinomialDistributionPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new MultinomialDistributionPlugin(name, serialData, serialLength);
}

void MultinomialDistributionPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *MultinomialDistributionPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *MultinomialDistributionPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *MultinomialDistributionPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

const PluginFieldCollection *MultinomialDistributionPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(MultinomialDistributionPluginCreator);

} // namespace nvinfer1