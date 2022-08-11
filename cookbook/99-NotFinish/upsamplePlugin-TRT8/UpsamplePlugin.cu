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

#include "UpsamplePlugin.h"

// 用于计算的 kernel
static __device__ float Bilinear(const float *pSrc, const int nSrcPitch, const int nSrcWidth, const int nSrcHeight, float x, float y)
{
    x = min(max(x - 0.5f, 0.0f), nSrcWidth - 1.0f);
    y = min(max(y - 0.5f, 0.0f), nSrcHeight - 1.0f);

    int x_low = (int)x;
    int y_low = (int)y;

    float lx = x - x_low;
    float ly = y - y_low;
    float hx = 1.0f - lx;
    float hy = 1.0f - ly;

    int x_high = min(x_low + 1, nSrcWidth - 1);
    int y_high = min(y_low + 1, nSrcHeight - 1);

    float v1 = *(float *)((uint8_t *)pSrc + y_low * nSrcPitch + x_low * sizeof(float));
    float v2 = *(float *)((uint8_t *)pSrc + y_low * nSrcPitch + x_high * sizeof(float));
    float v3 = *(float *)((uint8_t *)pSrc + y_high * nSrcPitch + x_low * sizeof(float));
    float v4 = *(float *)((uint8_t *)pSrc + y_high * nSrcPitch + x_high * sizeof(float));
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

static inline __device__ float Nearest(const float *pSrc, const int nSrcPitch, float x, float y)
{
    return *(float *)((uint8_t *)pSrc + int(y) * nSrcPitch + int(x) * sizeof(float));
}

static __global__ void Scale_fp32(bool bNearest, int nImage, float *pSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, float *pDst, int nDstPitch, int nDstWidth, int nDstHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nDstWidth || y >= nDstHeight)
    {
        return;
    }
    float fxScale = 1.0f * nSrcWidth / nDstWidth, fyScale = 1.0f * nSrcHeight / nDstHeight;
    for (int i = 0; i < nImage; i++)
    {
        *(float *)((uint8_t *)pDst + nDstPitch * y + x * sizeof(float)) =
            bNearest ? Nearest(pSrc, nSrcPitch, (x + 0.5f) * fxScale, (y + 0.5f) * fyScale) : Bilinear(pSrc, nSrcPitch, nSrcWidth, nSrcHeight, (x + 0.5f) * fxScale, (y + 0.5f) * fyScale);
        pSrc = (float *)((uint8_t *)pSrc + nSrcPitch * nSrcHeight);
        pDst = (float *)((uint8_t *)pDst + nDstPitch * nDstHeight);
    }
}

__global__ void resizeBilinearKernel(float *pInput, int nSrcWidth, int nSrcHeight, float *pOutput, int nDstWidth, int nDstHeight, float fx, float fy, int nBatch)
{
    int dstX = blockDim.x * blockIdx.x + threadIdx.x;
    int dstY = blockDim.y * blockIdx.y + threadIdx.y;

    if (dstX >= nDstWidth || dstY >= nDstHeight)
        return;

    int srcLeft, srcRght;
    int srcBotom, srcUpper;

    //int nFrame = blockDim.z;
    //added by chandler, 2020.3.22, optimize resize kernel
    int nFrame = blockIdx.z;

    float srcX = (dstX + 0.5f) * fx - 0.5f;
    float srcY = (dstY + 0.5f) * fy - 0.5f;
    // align with trt
    // float srcX = dstX * fx;
    // float srcY = dstY * fy;

    float valLeftUpper, valRghtUpper;
    float valLeftBotom, valRghtBotom;

    if (srcX < .0f)
        srcLeft = srcRght = 0;
    else if ((int)srcX >= nSrcWidth - 1)
        srcLeft = srcRght = nSrcWidth - 1;
    else
    {
        srcLeft = (int)srcX;
        srcRght = srcLeft + 1;
    }

    if (srcY < .0f)
        srcBotom = srcUpper = 0;
    else if ((int)srcY >= nSrcHeight - 1)
        srcBotom = srcUpper = nSrcHeight - 1;
    else
    {
        srcBotom = (int)srcY;
        srcUpper = srcBotom + 1;
    }

    float x = srcX - srcLeft;
    float y = srcY - srcBotom;

    int srcGid; // global id
    int srcGid_base = nFrame * nSrcHeight * nSrcWidth;

    srcGid       = srcGid_base + srcBotom * nSrcWidth + srcLeft;
    valLeftBotom = pInput[srcGid];

    srcGid       = srcGid_base + srcBotom * nSrcWidth + srcRght;
    valRghtBotom = pInput[srcGid];

    srcGid       = srcGid_base + srcUpper * nSrcWidth + srcLeft;
    valLeftUpper = pInput[srcGid];

    srcGid       = srcGid_base + srcUpper * nSrcWidth + srcRght;
    valRghtUpper = pInput[srcGid];

    pOutput[nFrame * nDstHeight * nDstWidth +
            dstY * nDstWidth +
            dstX] =
        (1 - x) * (1 - y) * valLeftBotom +
        (1 - x) * y * valLeftUpper +
        x * (1 - y) * valRghtBotom +
        x * y * valRghtUpper;

    return;
}

namespace nvinfer1
{
// 这里各成员函数按照被调用顺序或重要程度顺序排列
// class UpsamplePlugin
UpsamplePlugin::UpsamplePlugin(const std::string &name, int nScaleFactor, int bNearest):
    name_(name)
{
    WHERE_AM_I();
    m_.nScaleFactor = nScaleFactor;
    m_.bNearest     = bNearest;
}

UpsamplePlugin::UpsamplePlugin(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

UpsamplePlugin::~UpsamplePlugin()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *UpsamplePlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new UpsamplePlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t UpsamplePlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType UpsamplePlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

DimsExprs UpsamplePlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs ret(inputs[0]);
    ret.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *exprBuilder.constant(m_.nScaleFactor));
    ret.d[3] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[3], *exprBuilder.constant(m_.nScaleFactor));
    return ret;
}

bool UpsamplePlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
#ifdef DEBUG
    bool res;
    switch (pos)
    {
    case 0:
        res = inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
        break;
    case 1:
        res = inOut[1].format == inOut[0].format && inOut[1].type == inOut[0].type;
        break;
    default: // should NOT be here!
        res = false;
    }

    std::cout << "\tpos=" << pos << ",res=" << res << "->[";
    for (int i = 0; i < nbInputs + nbOutputs; ++i)
    {
        std::cout << getFormatString(inOut[i].format) << ",";
    }
    std::cout << "],[";
    for (int i = 0; i < nbInputs + nbOutputs; ++i)
    {
        std::cout << getDataTypeString(inOut[i].type) << ",";
    }
    std::cout << "]" << std::endl;
    return res;
#else
    switch (pos)
    {
    case 0:
        return inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
    default: // should NOT be here!
        return false;
    }
    return false;
#endif
}

void UpsamplePlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return;
}

size_t UpsamplePlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t UpsamplePlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();

    dim3 dimBlock;
    dim3 dimGrid;

    int nLeadingDimension = outputDesc[0].dims.d[0] * outputDesc[0].dims.d[1];
    int nSrcHeight        = inputDesc[0].dims.d[2];
    int nSrcWidth         = inputDesc[0].dims.d[3];
    int nDstHeight        = outputDesc[0].dims.d[2];
    int nDstWidth         = outputDesc[0].dims.d[3];
    dimBlock.x            = 16;
    dimBlock.y            = 16;
    dimGrid.x             = CEIL_DIVIDE(nDstWidth, dimBlock.x);
    dimGrid.y             = CEIL_DIVIDE(nDstHeight, dimBlock.y);
    dimGrid.z             = nLeadingDimension;
    float fx              = 1 / (float)m_.nScaleFactor;
    float fy              = 1 / (float)m_.nScaleFactor;

    resizeBilinearKernel<<<dimGrid, dimBlock, 0, stream>>>((float *)inputs[0],
                                                           nSrcWidth,
                                                           nSrcHeight,
                                                           (float *)outputs[0],
                                                           nDstWidth,
                                                           nDstHeight,
                                                           fx,
                                                           fy,
                                                           dimGrid.z); // batchsize * nChannel

    return 0;
}

void UpsamplePlugin::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t UpsamplePlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void UpsamplePlugin::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t UpsamplePlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void UpsamplePlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void UpsamplePlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *UpsamplePlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *UpsamplePlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *UpsamplePlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

void UpsamplePlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void UpsamplePlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class UpsamplePluginCreator
PluginFieldCollection    UpsamplePluginCreator::fc_ {};
std::vector<PluginField> UpsamplePluginCreator::attr_;

UpsamplePluginCreator::UpsamplePluginCreator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("nScaleFactor", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("bNearest", nullptr, PluginFieldType::kINT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

UpsamplePluginCreator::~UpsamplePluginCreator()
{
    WHERE_AM_I();
}

// 最重要的两个成员函数，分别用于“接受参数创建 Plugin” 和 “去序列化创建 Plugin”
IPluginV2 *UpsamplePluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    int                          nScaleFactor = 0;
    int                          bNearest     = 0;
    std::map<std::string, int *> parameterMap {{"nScaleFactor", &nScaleFactor}, {"bNearest", &bNearest}};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
    }
    UpsamplePlugin *pObj = new UpsamplePlugin(name, nScaleFactor, bNearest);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2 *UpsamplePluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    UpsamplePlugin *pObj = new UpsamplePlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void UpsamplePluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *UpsamplePluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *UpsamplePluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *UpsamplePluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

const PluginFieldCollection *UpsamplePluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(UpsamplePluginCreator);

} // namespace nvinfer1
