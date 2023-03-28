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

#include "Resize2DPlugin.h"

// 用于计算的 kernel
template<typename T>
__global__ void nearestResize2DV1(const T *pInput, T *pOutput, const int nBC, const int nH0, const int nW0, const int nH1, const int nW1)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nH1 || col >= nW1)
        return;

    float alpha = (row + 0.5f) * nH0 / nH1;
    float beta  = (col + 0.5f) * nW0 / nW1;
    int   srcU  = (int)alpha;
    int   srcL  = (int)beta;

    for (int i = 0; i < nBC; ++i) // nB * nC
    {
        int srcIndex      = i * nH0 * nW0 + srcU * nW0 + srcL;
        int dstIndex      = i * nH1 * nW1 + row * nW1 + col;
        pOutput[dstIndex] = pInput[srcIndex];
    }
    return;
}

template __global__ void nearestResize2DV1<float>(const float *, float *, const int, const int, const int, const int, const int);
template __global__ void nearestResize2DV1<half>(const half *, half *, const int, const int, const int, const int, const int);

template<typename T, typename OP_T> // 输入输出数据类型和计算数据类型
__global__ void bilinearResize2DV1(const T *pInput, T *pOutput, const int nBC, const int nH0, const int nW0, const int nH1, const int nW1)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nH1 || col >= nW1)
        return;

    float alpha = min(max((row + 0.5f) * nH0 / nH1 - 0.5f, 0.0f), nH0 - 1.0f); // alpha 和 beta 的计算方法参见 02-API/Layer/ResizeLayer/Resize层.md
    float beta  = min(max((col + 0.5f) * nW0 / nW1 - 0.5f, 0.0f), nW0 - 1.0f);
    int   srcU  = (int)alpha;
    int   srcD  = min(srcU + 1, nH0 - 1);
    int   srcL  = (int)beta;
    int   srcR  = min(srcL + 1, nW0 - 1);
    alpha       = alpha - (int)alpha;
    beta        = beta - (int)beta;

    for (int i = 0; i < nBC; ++i)
    {
        OP_T vUL          = pInput[i * nH0 * nW0 + srcU * nW0 + srcL];
        OP_T vUR          = pInput[i * nH0 * nW0 + srcU * nW0 + srcR];
        OP_T vDL          = pInput[i * nH0 * nW0 + srcD * nW0 + srcL];
        OP_T vDR          = pInput[i * nH0 * nW0 + srcD * nW0 + srcR];
        int  dstIndex     = i * nH1 * nW1 + row * nW1 + col;
        pOutput[dstIndex] = T(vUL * OP_T(1 - alpha) * OP_T(1 - beta) +
                              vUR * OP_T(1 - alpha) * OP_T(beta) +
                              vDL * OP_T(alpha) * OP_T(1 - beta) +
                              vDR * OP_T(alpha) * OP_T(beta));
    }
    return;
}

template __global__ void bilinearResize2DV1<float, float>(const float *, float *, const int, const int, const int, const int, const int);
template __global__ void bilinearResize2DV1<half, float>(const half *, half *, const int, const int, const int, const int, const int);
template __global__ void bilinearResize2DV1<half, half>(const half *, half *, const int, const int, const int, const int, const int);

template<typename T>
__global__ void nearestResize2DV2(const T *pInput, T *pOutput, const int nB, const int nC, const int nH0, const int nW0, const int nH1, const int nW1)
{
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nH1 || col >= nW1 * nC)
        return;

    float alpha = (row + 0.5f) * nH0 / nH1;
    float beta  = (col / nC + 0.5f) * nW0 / nW1;
    int   srcU  = (int)alpha;
    int   srcL  = (int)beta;
    int   iC    = col % nC;

    for (int i = 0; i < nB; ++i)
    {
        int srcIndex      = i * nH0 * nW0 * nC + srcU * nW0 * nC + srcL * nC + iC;
        int dstIndex      = i * nH1 * nW1 * nC + row * nW1 * nC + col;
        pOutput[dstIndex] = pInput[srcIndex];
    }
    return;
}

template __global__ void nearestResize2DV2<float>(const float *, float *, const int, const int, const int, const int, const int, const int);
template __global__ void nearestResize2DV2<half>(const half *, half *, const int, const int, const int, const int, const int, const int);

template<typename T, typename OP_T> // 输入输出数据类型和计算数据类型
__global__ void bilinearResize2DV2(const T *pInput, T *pOutput, const int nB, const int nC, const int nH0, const int nW0, const int nH1, const int nW1)
{
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nH1 || col >= nW1 * nC)
        return;

    float alpha = min(max((row + 0.5f) * nH0 / nH1 - 0.5f, 0.0f), nH0 - 1.0f); // alpha 和 beta 的计算方法参见 02-API/Layer/ResizeLayer/Resize层.md
    float beta  = min(max((col / nC + 0.5f) * nW0 / nW1 - 0.5f, 0.0f), nW0 - 1.0f);
    int   srcU  = (int)alpha;
    int   srcD  = min(srcU + 1, nH0 - 1);
    int   srcL  = (int)beta;
    int   srcR  = min(srcL + 1, nW0 - 1);
    int   iC    = col % nC;
    alpha       = alpha - (int)alpha;
    beta        = beta - (int)beta;

    for (int i = 0; i < nB; ++i)
    {
        OP_T vUL          = pInput[i * nH0 * nW0 * nC + (srcU * nW0 + srcL) * nC + iC];
        OP_T vUR          = pInput[i * nH0 * nW0 * nC + (srcU * nW0 + srcR) * nC + iC];
        OP_T vDL          = pInput[i * nH0 * nW0 * nC + (srcD * nW0 + srcL) * nC + iC];
        OP_T vDR          = pInput[i * nH0 * nW0 * nC + (srcD * nW0 + srcR) * nC + iC];
        int  dstIndex     = i * nH1 * nW1 * nC + row * nW1 * nC + col;
        pOutput[dstIndex] = T(vUL * OP_T(1 - alpha) * OP_T(1 - beta) +
                              vUR * OP_T(1 - alpha) * OP_T(beta) +
                              vDL * OP_T(alpha) * OP_T(1 - beta) +
                              vDR * OP_T(alpha) * OP_T(beta));
    }
    return;
}

template __global__ void bilinearResize2DV2<float, float>(const float *, float *, const int, const int, const int, const int, const int, const int);
template __global__ void bilinearResize2DV2<half, float>(const half *, half *, const int, const int, const int, const int, const int, const int);
template __global__ void bilinearResize2DV2<half, half>(const half *, half *, const int, const int, const int, const int, const int, const int);

namespace nvinfer1
{
// class Resize2DPluginV1
Resize2DPluginV1::Resize2DPluginV1(const std::string &name, int nMode, int nScale, int nH1, int nW1):
    name_(name)
{
    WHERE_AM_I();
    m_.nMode  = nMode;
    m_.nScale = nScale;
    m_.nH1    = nH1;
    m_.nW1    = nW1;
}

Resize2DPluginV1::Resize2DPluginV1(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

Resize2DPluginV1::~Resize2DPluginV1()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *Resize2DPluginV1::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new Resize2DPluginV1(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t Resize2DPluginV1::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType Resize2DPluginV1::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

DimsExprs Resize2DPluginV1::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs ret = inputs[0];
    if (m_.nScale > 0 && m_.nH1 == 0 && m_.nW1 == 0)
    {
        ret.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *exprBuilder.constant(m_.nScale));
        ret.d[3] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[3], *exprBuilder.constant(m_.nScale));
    }
    else if (m_.nScale <= 0 && m_.nH1 > 0 && m_.nW1 > 0)
    {
        ret.d[2] = exprBuilder.constant(m_.nH1);
        ret.d[3] = exprBuilder.constant(m_.nW1);
    }
    else if (m_.nScale > 0 && (m_.nH1 > 0 || m_.nW1 > 0))
    {
        std::cout << "[Resize2DPluginV1::getOutputDimensions]Do not use both SCALAR and OUTPUT_SIZE" << std::endl;
    }
    else
    {
        std::cout << "[Resize2DPluginV1::getOutputDimensions]Error setting output size" << std::endl;
    }
    return ret;
}

bool Resize2DPluginV1::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    switch (pos)
    {
    case 0:
        return (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF) && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
    default: // should NOT be here!
        return false;
    }
    return false;
}

void Resize2DPluginV1::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    if (in->desc.dims.d[0] > 0 && m_.nScale > 0 && m_.nH1 == 0 && m_.nW1 == 0) // Runtime and use Scale
    {
        m_.nH1 = in->desc.dims.d[2] * m_.nScale;
        m_.nW1 = in->desc.dims.d[3] * m_.nScale;
    }
    return;
}

size_t Resize2DPluginV1::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t Resize2DPluginV1::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    m_.nH0 = inputDesc[0].dims.d[2];
    m_.nW0 = inputDesc[0].dims.d[3];

    dim3 block(16, 16);
    dim3 grid(CEIL_DIVIDE(m_.nW1, 16), CEIL_DIVIDE(m_.nH1, 16));

    switch (int(inputDesc[0].type) * 4 + m_.nMode)
    {
    case 0: // kFLOAT, nearest interpolation
        (nearestResize2DV1<float>)<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float *>(inputs[0]),
            reinterpret_cast<float *>(outputs[0]),
            inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1],
            m_.nH0,
            m_.nW0,
            m_.nH1,
            m_.nW1);
        break;
    case 1: // kFLOAT, bilinear interpolation
        (bilinearResize2DV1<float, float>)<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float *>(inputs[0]),
            reinterpret_cast<float *>(outputs[0]),
            inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1],
            m_.nH0,
            m_.nW0,
            m_.nH1,
            m_.nW1);
        break;
    case 4: // kHALF, nearest interpolation
        (nearestResize2DV1<float>)<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float *>(inputs[0]),
            reinterpret_cast<float *>(outputs[0]),
            inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1],
            m_.nH0,
            m_.nW0,
            m_.nH1,
            m_.nW1);
        break;
    case 5: // kHALF, bilinear interpolation
        (bilinearResize2DV1<half, float>)<<<grid, block, 0, stream>>>(
            reinterpret_cast<const half *>(inputs[0]),
            reinterpret_cast<half *>(outputs[0]),
            inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1],
            m_.nH0,
            m_.nW0,
            m_.nH1,
            m_.nW1);
        break;
    default:
        std::cout << std::string("[Resize2DPluginV1::enqueue]Error combination of DataType and InterpolationMode!") << std::endl;
    }
    return 0;
}

void Resize2DPluginV1::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t Resize2DPluginV1::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void Resize2DPluginV1::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t Resize2DPluginV1::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void Resize2DPluginV1::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void Resize2DPluginV1::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *Resize2DPluginV1::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *Resize2DPluginV1::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *Resize2DPluginV1::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION_V1;
}

void Resize2DPluginV1::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void Resize2DPluginV1::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class Resize2DPluginV1Creator
PluginFieldCollection    Resize2DPluginV1Creator::fc_ {};
std::vector<PluginField> Resize2DPluginV1Creator::attr_;

Resize2DPluginV1Creator::Resize2DPluginV1Creator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("Mode", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("Scale", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("OutputHeight", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("OutputWidth", nullptr, PluginFieldType::kINT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}
Resize2DPluginV1Creator::~Resize2DPluginV1Creator()
{
    WHERE_AM_I();
}

IPluginV2 *Resize2DPluginV1Creator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    int                          nMode  = 0;
    int                          nScale = 0;
    int                          nH1    = 0;
    int                          nW1    = 0;
    std::map<std::string, int *> parameterMap;
    parameterMap["Mode"]         = &nMode;
    parameterMap["Scale"]        = &nScale;
    parameterMap["OutputHeight"] = &nH1;
    parameterMap["OutputWidth"]  = &nW1;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
    }
    Resize2DPluginV1 *pObj = new Resize2DPluginV1(name, nMode, nScale, nH1, nW1);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2 *Resize2DPluginV1Creator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    Resize2DPluginV1 *pObj = new Resize2DPluginV1(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void Resize2DPluginV1Creator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *Resize2DPluginV1Creator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *Resize2DPluginV1Creator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *Resize2DPluginV1Creator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION_V1;
}

const PluginFieldCollection *Resize2DPluginV1Creator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(Resize2DPluginV1Creator);

// class Resize2DPluginV2
Resize2DPluginV2::Resize2DPluginV2(const std::string &name, int nMode, int nScale, int nH1, int nW1):
    name_(name)
{
    WHERE_AM_I();
    m_.nMode  = nMode;
    m_.nScale = nScale;
    m_.nH1    = nH1;
    m_.nW1    = nW1;
}

Resize2DPluginV2::Resize2DPluginV2(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

Resize2DPluginV2::~Resize2DPluginV2()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *Resize2DPluginV2::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new Resize2DPluginV2(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t Resize2DPluginV2::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType Resize2DPluginV2::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

DimsExprs Resize2DPluginV2::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs ret = inputs[0];
    if (m_.nScale > 0 && m_.nH1 == 0 && m_.nW1 == 0)
    {
        ret.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *exprBuilder.constant(m_.nScale));
        ret.d[3] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[3], *exprBuilder.constant(m_.nScale));
    }
    else if (m_.nScale <= 0 && m_.nH1 > 0 && m_.nW1 > 0)
    {
        ret.d[2] = exprBuilder.constant(m_.nH1);
        ret.d[3] = exprBuilder.constant(m_.nW1);
    }
    else if (m_.nScale > 0 && (m_.nH1 > 0 || m_.nW1 > 0))
    {
        std::cout << "[Resize2DPluginV2::getOutputDimensions]Do not use both SCALAR and OUTPUT_SIZE" << std::endl;
    }
    else
    {
        std::cout << "[Resize2DPluginV2::getOutputDimensions]Error setting output size" << std::endl;
    }
    return ret;
}

bool Resize2DPluginV2::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
#ifdef DEBUG
    bool res;
    switch (pos)
    {
    case 0:
        res = inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kHWC ||
              inOut[0].type == DataType::kHALF && inOut[0].format == TensorFormat::kHWC8;
        break;
    case 1:
        res = inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
        break;
    default: // should NOT be here!
        res = false;
    }

    std::cout << "\tpos=" << pos << ",res=" << res << "->[";
    for (int i = 0; i < nbInputs + nbOutputs; ++i)
    {
        std::cout << formatToString(inOut[i].format) << ",";
    }
    std::cout << "],[";
    for (int i = 0; i < nbInputs + nbOutputs; ++i)
    {
        std::cout << dataTypeToString(inOut[i].type) << ",";
    }
    std::cout << "]" << std::endl;
    return res;
#else
    switch (pos)
    {
    case 0:
        return inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kHWC ||
               inOut[0].type == DataType::kHALF && inOut[0].format == TensorFormat::kHWC8;
    case 1:
        return inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
    default: // should NOT be here!
        return false;
    }
    return false;
#endif
}

void Resize2DPluginV2::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    if (in->desc.dims.d[0] > 0 && m_.nScale > 0 && m_.nH1 == 0 && m_.nW1 == 0) // Runtime and use Scale
    {
        m_.nH1 = in->desc.dims.d[2] * m_.nScale;
        m_.nW1 = in->desc.dims.d[3] * m_.nScale;
    }
    return;
}

size_t Resize2DPluginV2::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t Resize2DPluginV2::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    const int nB = inputDesc[0].dims.d[0];
    const int nC = inputDesc[0].dims.d[1];
    m_.nH0       = inputDesc[0].dims.d[2];
    m_.nW0       = inputDesc[0].dims.d[3];

    constexpr int TPB = 256;
    dim3          block(TPB, 1, 1);
    dim3          grid(CEIL_DIVIDE(m_.nW1 * nC, TPB), m_.nH1, 1);

    switch (int(inputDesc[0].type) * 4 + m_.nMode)
    {
    case 0: // kFLOAT, nearest interpolation
        (nearestResize2DV2<float>)<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float *>(inputs[0]),
            reinterpret_cast<float *>(outputs[0]),
            nB,
            nC,
            m_.nH0,
            m_.nW0,
            m_.nH1,
            m_.nW1);
        break;
    case 1: // kFLOAT, bilinear interpolation
        (bilinearResize2DV2<float, float>)<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float *>(inputs[0]),
            reinterpret_cast<float *>(outputs[0]),
            nB,
            nC,
            m_.nH0,
            m_.nW0,
            m_.nH1,
            m_.nW1);
        break;
    case 4: // kHALF, nearest interpolation
        (nearestResize2DV2<float>)<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float *>(inputs[0]),
            reinterpret_cast<float *>(outputs[0]),
            nB,
            nC,
            m_.nH0,
            m_.nW0,
            m_.nH1,
            m_.nW1);
        break;
    case 5: // kHALF, bilinear interpolation
        (bilinearResize2DV2<half, float>)<<<grid, block, 0, stream>>>(
            reinterpret_cast<const half *>(inputs[0]),
            reinterpret_cast<half *>(outputs[0]),
            nB,
            nC,
            m_.nH0,
            m_.nW0,
            m_.nH1,
            m_.nW1);
        break;
    default:
        std::cout << std::string("[Resize2DPluginV2::enqueue]Error combination of DataType and InterpolationMode!") << std::endl;
    }
    return 0;
}

void Resize2DPluginV2::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t Resize2DPluginV2::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void Resize2DPluginV2::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t Resize2DPluginV2::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void Resize2DPluginV2::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void Resize2DPluginV2::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *Resize2DPluginV2::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *Resize2DPluginV2::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *Resize2DPluginV2::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION_V2;
}

void Resize2DPluginV2::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void Resize2DPluginV2::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class Resize2DPluginV2Creator
PluginFieldCollection    Resize2DPluginV2Creator::fc_ {};
std::vector<PluginField> Resize2DPluginV2Creator::attr_;

Resize2DPluginV2Creator::Resize2DPluginV2Creator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("Mode", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("Scale", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("OutputHeight", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("OutputWidth", nullptr, PluginFieldType::kINT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}
Resize2DPluginV2Creator::~Resize2DPluginV2Creator()
{
    WHERE_AM_I();
}

IPluginV2 *Resize2DPluginV2Creator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    int                          nMode  = 0;
    int                          nScale = 0;
    int                          nH1    = 0;
    int                          nW1    = 0;
    std::map<std::string, int *> parameterMap;
    parameterMap["Mode"]         = &nMode;
    parameterMap["Scale"]        = &nScale;
    parameterMap["OutputHeight"] = &nH1;
    parameterMap["OutputWidth"]  = &nW1;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
    }
    Resize2DPluginV2 *pObj = new Resize2DPluginV2(name, nMode, nScale, nH1, nW1);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2 *Resize2DPluginV2Creator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    Resize2DPluginV2 *pObj = new Resize2DPluginV2(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void Resize2DPluginV2Creator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *Resize2DPluginV2Creator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *Resize2DPluginV2Creator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *Resize2DPluginV2Creator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION_V2;
}

const PluginFieldCollection *Resize2DPluginV2Creator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(Resize2DPluginV2Creator);

} // namespace nvinfer1