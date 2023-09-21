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

#include "fMHAPlugin.h"

__global__ void printGPUHalf(const half *in, const int n)
{
    printf("\n");
    for (int i = 0; i < n; ++i)
    {
        printf("%d: %f\n", i, float(in[i]));
    }
    printf("\n");
    return;
}

__global__ void printGPUFloat(float *in, const int n)
{
    printf("\n");
    for (int i = 0; i < n; ++i)
    {
        printf("%d: %f\n", i, in[i]);
    }
    printf("\n");
    return;
}

__global__ void printGPUInt(int *in, const int n)
{
    printf("\n");
    for (int i = 0; i < n; ++i)
    {
        printf("%d: %d\n", i, in[i]);
    }
    printf("\n");
    return;
}

namespace nvinfer1
{
// class fMHAPlugin
fMHAPlugin::fMHAPlugin(const std::string &name, float dropout_p, float scale, int causal, int return_attn_probs):
    name_(name)
{
    WHERE_AM_I();
    m_.dropout_p           = dropout_p;
    m_.scale           = scale;
    m_.causal           = causal;
    m_.return_attn_probs           = return_attn_probs;
    init();
}

fMHAPlugin::fMHAPlugin(const std::string &name,
                                 const void *       buffer,
                                 size_t             length):
    name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

void fMHAPlugin::init()
{
    // initialize seqlens buffer
    allocateSeqlens(m_.mMaxBatchSize);
    m_.mOptSeqLenQ
        = initializeSeqlens(m_.mOptBatchSize, m_.mOptSeqLenQ, mCuSeqLensQ.get());
    m_.mOptSeqLenKV
        = initializeSeqlens(m_.mOptBatchSize, m_.mOptSeqLenKV, mCuSeqLensKV.get());
}

fMHAPlugin::~fMHAPlugin()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *fMHAPlugin::clone() const noexcept
{
    WHERE_AM_I();
    std::vector<char> buff;
    buff.resize(getSerializationSize());
    serialize(buff.data());

    auto p = new fMHAPlugin(name_, buff.data(), buff.size());
    p->mCuSeqLensQ = mCuSeqLensQ;
    p->mCuSeqLensKV = mCuSeqLensKV;
    p->setPluginNamespace(namespace_.c_str());
    p->init();
    return p;
}

int32_t fMHAPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType fMHAPlugin::getOutputDataType(int32_t         index,
                                        DataType const *inputTypes,
                                        int32_t         nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

DimsExprs fMHAPlugin::getOutputDimensions(
    int32_t          outputIndex,
    const DimsExprs *inputs,
    int32_t          nbInputs,
    IExprBuilder &   exprBuilder) noexcept
{
    WHERE_AM_I();
    return inputs[0];
}

bool fMHAPlugin::supportsFormatCombination(int32_t                 pos,
                                            const PluginTensorDesc *inOut,
                                            int32_t                 nbInputs,
                                            int32_t                 nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res;
    switch (pos)
    {
    case 0:
    case 1:
    case 2:
    case 3:
        res = (inOut[pos].type == DataType::kHALF) && inOut[pos].format == TensorFormat::kLINEAR;
        break;
    // case 3:
    // case 4:
    //     res = (inOut[pos].type == DataType::kINT32) && inOut[pos].format == TensorFormat::kLINEAR;
    //     break;
    default: // should NOT be here!
        res = false;
    }
    return res;
    /*
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
    */
}

void fMHAPlugin::allocateSeqlens(int32_t maxBatchSize)
{
    // allocate seqlens buffer
    auto allocBuffer = [&maxBatchSize](cuda_shared_ptr<void>& dptr) {
        if (!dptr && maxBatchSize)
        {
            void* cudaMem{nullptr};
            cudaMalloc(&cudaMem, sizeof(int32_t) * (maxBatchSize + 1));
            make_cuda_shared(dptr, cudaMem);
        }
    };
    allocBuffer(mCuSeqLensQ);
    allocBuffer(mCuSeqLensKV);
    m_.mMaxBatchSize = maxBatchSize;
}

int32_t fMHAPlugin::initializeSeqlens(int32_t b, int32_t s, void* cuSeqlensDev, cudaStream_t stream)
{
    if (!b || !s)
    {
        return s;
    }

    std::vector<int32_t> cuSeqlens(b + 1, 0);
    // Compute the prefix sum of the sequence lenghts.
    for (int32_t it = 0; it < b; it++)
    {
        cuSeqlens[it + 1] = cuSeqlens[it] + s;
    }

    cudaMemcpyAsync(
        cuSeqlensDev, cuSeqlens.data(), sizeof(int32_t) * cuSeqlens.size(), cudaMemcpyHostToDevice, stream);
    m_.mOptBatchSize = b;
    return s;
}

void fMHAPlugin::configurePlugin(const DynamicPluginTensorDesc *in,
                                      int32_t                        nbInputs,
                                      const DynamicPluginTensorDesc *out,
                                      int32_t                        nbOutputs) noexcept
{
    WHERE_AM_I();
    int32_t const batchSize = in[0].max.d[0];
    int32_t const seqLenQ = in[0].max.d[1];
    int32_t const seqLenKV = in[1].max.d[1];

    allocateSeqlens(batchSize);
    if (batchSize != m_.mOptBatchSize || m_.mOptSeqLenQ != seqLenQ
        || m_.mOptSeqLenKV != seqLenKV)
    {
        m_.mOptSeqLenQ = initializeSeqlens(batchSize, seqLenQ, mCuSeqLensQ.get());
        m_.mOptSeqLenKV = initializeSeqlens(batchSize, seqLenKV, mCuSeqLensKV.get());
    }

    m_.mDataType = in[0].desc.type;
    return;
}

size_t fMHAPlugin::getWorkspaceSize(const PluginTensorDesc *inputs,
                                         int32_t                 nbInputs,
                                         const PluginTensorDesc *outputs,
                                         int32_t                 nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 2 << 28; // why 2 << 30 cause the illegal memory error??????
}

int32_t fMHAPlugin::enqueue(const PluginTensorDesc *inputDesc,
                                 const PluginTensorDesc *outputDesc,
                                 const void *const *     inputs,
                                 void *const *           outputs,
                                 void *                  workspace,
                                 cudaStream_t            stream) noexcept
{
#ifdef DEBUG
    nvtxRangePushA("fMHAPlugin");
#endif
    WHERE_AM_I();
    // Q: [bs, q_seqlen, num_heads, head_size]
    // K: [bs, kv_seqlen, num_heads, head_size]
    // V: [bs, kv_seqlen, num_heads, head_size]
    // Output: [bs, q_seqlen, num_heads, head_size]

    int batch_size = inputDesc[0].dims.d[0];
    int q_seqlen = inputDesc[0].dims.d[1];
    int total_q = batch_size * q_seqlen;
    int num_heads = inputDesc[0].dims.d[2];
    int head_size = inputDesc[0].dims.d[3];
    int kv_seqlen = inputDesc[1].dims.d[1];
    int total_k = batch_size * kv_seqlen;
    
    if (batch_size != m_.mOptBatchSize || m_.mOptSeqLenQ != q_seqlen
        || m_.mOptSeqLenKV != kv_seqlen)
    {
        m_.mOptSeqLenQ = initializeSeqlens(batch_size, q_seqlen, mCuSeqLensQ.get(), stream);
        m_.mOptSeqLenKV = initializeSeqlens(batch_size, kv_seqlen, mCuSeqLensKV.get(), stream);
    }
    
    // printf("\ntotal_q=%d, batch_size=%d, num_heads=%d, head_size=%d, total_k=%d\n", total_q, batch_size, num_heads, head_size, total_k);
    // printf("\nq_seqlen=%d, kv_seqlen=%d, dropout_p=%f, scale=%f, causal=%d, return_attn_probs=%d\n", q_seqlen, m_.kv_seqlen, m_.dropout_p, m_.scale, m_.causal, m_.return_attn_probs);
    // printGPUInt<<<1, 1>>>((int*)(mCuSeqLensQ.get()), batch_size + 1);
    // printGPUInt<<<1, 1>>>((int*)(mCuSeqLensKV.get()), batch_size + 1);
    // printGPUHalf<<<1, 1>>>((half*)(inputs[0]), 3 * num_heads * head_size);
    // printGPUHalf<<<1, 1>>>((half*)(inputs[1]), 3 * num_heads * head_size);
    // printGPUHalf<<<1, 1>>>((half*)(inputs[2]), 3 * num_heads * head_size);
    mha_fwd(
        (half*)(inputs[0]),         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        (half*)(inputs[1]),         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        (half*)(inputs[2]),         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        (half*)(outputs[0]),        // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        (void*)(mCuSeqLensQ.get()),         // b+1
        (void*)(mCuSeqLensKV.get()),         // b+1
        (float*)workspace,
        q_seqlen,
        kv_seqlen,
        batch_size,
        total_q,
        num_heads,
        head_size,
        total_k,
        m_.dropout_p,
        m_.scale,
        false,                      // zero_tensors
        (bool)(m_.causal),
        (bool)(m_.return_attn_probs),
        0,                          // num_splits
        stream);
    // printf("-------Check plugin output data: \n");
    // printGPUHalf1<<<1, 1>>>((half*)(outputs[0]), batch_size * q_seqlen * num_heads * head_size);

    // Make sure it launched ok.
    CHECK(cudaGetLastError());
#ifdef DEBUG
    nvtxRangePop();
#endif
    return 0;
}

void fMHAPlugin::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

int32_t fMHAPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void fMHAPlugin::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t fMHAPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void fMHAPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void fMHAPlugin::setPluginNamespace(
    const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *fMHAPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *fMHAPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *fMHAPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

void fMHAPlugin::attachToContext(cudnnContext * contextCudnn,
                                      cublasContext *contextCublas,
                                      IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void fMHAPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class fMHAPluginCreator
PluginFieldCollection    fMHAPluginCreator::fc_ {};
std::vector<PluginField> fMHAPluginCreator::attr_;

fMHAPluginCreator::fMHAPluginCreator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("dropout_p", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("causal", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("return_attn_probs", nullptr, PluginFieldType::kINT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

fMHAPluginCreator::~fMHAPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2 *fMHAPluginCreator::createPlugin(
    const char *                 name,
    const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    float dropout_p = 0.f;
    float scale = 1.f;
    int causal = 0;
    int return_attn_probs = 0;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (fc->fields[i].name == std::string("dropout_p"))
        {
            dropout_p = *reinterpret_cast<const float *>(fc->fields[i].data);
            continue;
        }
        if (fc->fields[i].name == std::string("scale"))
        {
            scale = *reinterpret_cast<const float *>(fc->fields[i].data);
            continue;
        }
        if (fc->fields[i].name == std::string("causal"))
        {
            causal = *reinterpret_cast<const int *>(fc->fields[i].data);
            continue;
        }
        if (fc->fields[i].name == std::string("return_attn_probs"))
        {
            return_attn_probs = *reinterpret_cast<const int *>(fc->fields[i].data);
            continue;
        }
    }
    return new fMHAPlugin(name, dropout_p, scale, causal, return_attn_probs);
}

IPluginV2 *fMHAPluginCreator::deserializePlugin(
    const char *name,
    const void *serialData,
    size_t      serialLength) noexcept
{
    WHERE_AM_I();
    auto p = new fMHAPlugin(name, serialData, serialLength);
    p->init();
    return p;
}

void fMHAPluginCreator::setPluginNamespace(
    const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *fMHAPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *fMHAPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *fMHAPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

const PluginFieldCollection *
fMHAPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(fMHAPluginCreator);

} // namespace nvinfer1
