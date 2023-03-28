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

#include "GruPlugin.h"

using namespace nvinfer1;

//PluginFieldCollection GruPluginCreator::mFC{};
//std::vector<PluginField> GruPluginCreator::mPluginAttributes;

namespace
{
const char *GRU_PLUGIN_VERSION {"0"};
const char *GRU_PLUGIN_NAME {"GruPlugin"};
const int   GRU_PLUGIN_NUM_INPUT  = 2;
const int   GRU_PLUGIN_NUM_OUTPUT = 2;
} // namespace

// activation functions
__device__ __forceinline__ float sigmoidActivation(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float tanhActivation(float x)
{
    return tanhf(x);
}

// for fp16 operations
template<typename T>
__device__ __forceinline__ T fp_mul(T a, T b);
template<>
__device__ __forceinline__ float fp_mul<float>(float a, float b)
{
    return (a * b);
};
template<>
__device__ __forceinline__ __half fp_mul<__half>(__half a, __half b)
{
    return __hmul(a, b);
};

template<typename TypeIn, typename TypeOut>
__device__ __forceinline__ TypeOut fp_convert(TypeIn in);
template<>
__device__ __forceinline__ float fp_convert<float, float>(float in)
{
    return in;
};
template<>
__device__ __forceinline__ float fp_convert<__half, float>(__half in)
{
    return __half2float(in);
};
template<>
__device__ __forceinline__ __half fp_convert<float, __half>(float in)
{
    return __float2half(in);
};
template<>
__device__ __forceinline__ __half fp_convert<__half, __half>(__half in)
{
    return in;
};

// GRU kernel implementation: exclude input matrix multiply
// early stop method: only get the final state of each sample (the input matrix is still zero-padding)
template<typename T>
__global__ void gruCellPersistKernel(
    int batch_size,
    int input_size,
    int hidden_size,
    int max_seq_len,
    const T *__restrict__ mat_wx,  // shape=[batch_size, max_seq_len, hidden_size*3].
                                   // The result of input matrix multiply: X*W.
                                   // It represents: [[X * Wxz], [X * Wxr], [X * Wxh]]
    const int *seq_len_arr,        // shape=[batch_size]. The sequence length of each sample.
    const T *__restrict__ weights, // shape=[hidden_size, hidden_size*3].
                                   // It represents: [[Whz], [Whr], [Whh]]
    const T *__restrict__ bias,    // shape=[hidden_size*3, 1].
                                   // It represents: [[bz], [br], [bh]]
    T *pre_state_cell,             // shape=[batch_size, hidden_size].
                                   // The temp buffer for storing pre-cell hidden state
    T *rh_cell,                    // shape=[batch_size, hidden_size].
                                   // The temp buffer for storing rh results
    T *outputs,                    // shape=[batch_size, max_seq_len, hidden_size]
    T *outputs_final               // shape=[batch_size, hidden_size]. Only get the final state of each sample
)
{
    int col     = blockIdx.x * blockDim.x + threadIdx.x; // dim=hidden_size
    int row     = blockIdx.y * blockDim.y + threadIdx.y; // dim=batch_size
    int seq_len = seq_len_arr[row];                      // the seq_len of each sample

    if ((row < batch_size) && (col < hidden_size))
    {
        int weights_hz_ind = 0; // 3 * input_size * hidden_size;
        int weights_hr_ind = weights_hz_ind + hidden_size * hidden_size;
        int weights_hh_ind = weights_hr_ind + hidden_size * hidden_size;

        int xz_idx = row * max_seq_len * hidden_size * 3 + col;
        int xr_idx = xz_idx + hidden_size;
        int xh_idx = xr_idx + hidden_size;

        int output_index = row * hidden_size + col;
        // each sample will be early stopped according to its seq_len
        for (int cell = 0; cell < seq_len; cell++)
        {
            T xz = mat_wx[xz_idx];
            T xr = mat_wx[xr_idx];
            T xh = mat_wx[xh_idx];

            xz_idx += hidden_size * 3;
            xr_idx += hidden_size * 3;
            xh_idx += hidden_size * 3;

            // compute hz, hr (gemm)w_index
            float hz = 0.0f, hr = 0.0f;
            int   h_index_base = row * hidden_size;
            int   w_index      = col;
            for (int i = 0; i < hidden_size; ++i)
            {
                T h = pre_state_cell[h_index_base + i];
                hz += fp_convert<T, float>(fp_mul<T>(h, weights[weights_hz_ind + w_index]));
                hr += fp_convert<T, float>(fp_mul<T>(h, weights[weights_hr_ind + w_index]));
                w_index += hidden_size;
            }

            // compute reset gate and update gate (element-wise)
            float z_gate = sigmoidActivation(fp_convert<T, float>(xz) + hz + fp_convert<T, float>(bias[col]));
            float r_gate = sigmoidActivation(fp_convert<T, float>(xr) + hr + fp_convert<T, float>(bias[col + hidden_size]));
            T     ht_1   = pre_state_cell[output_index]; // h(t-1)

            // compute r_gate(.)ht_1, and record the results in GMEM (element-wise)
            rh_cell[output_index] = fp_mul<T>(fp_convert<float, T>(r_gate), ht_1);

            // intra-block sync
            __syncthreads();

            // compute hh (gemm)
            float hh            = 0.0f;
            int   rh_index_base = row * hidden_size;
            w_index             = col;
            for (int i = 0; i < hidden_size; ++i)
            {
                T rh = rh_cell[rh_index_base + i];
                hh += fp_convert<T, float>(fp_mul<T>(rh, weights[weights_hh_ind + w_index]));
                w_index += hidden_size;
            }

            // compute candidate gate (element-wise)
            float h_gate = tanhActivation(fp_convert<T, float>(xh) + hh + fp_convert<T, float>(bias[col + hidden_size * 2]));

            // compute output state (element-wise)
            float output                                                        = z_gate * h_gate + (1.0f - z_gate) * fp_convert<T, float>(ht_1);
            outputs[row * max_seq_len * hidden_size + cell * hidden_size + col] = fp_convert<float, T>(output);

            // record the output hidden_state as next cell's pre_state
            pre_state_cell[output_index] = fp_convert<float, T>(output);

            // get the final state of each sample
            if (cell == (seq_len - 1))
            {
                outputs_final[output_index] = fp_convert<float, T>(output);
            }

            // intra-block sync
            __syncthreads();
        }
    }
}

// convert fp32 to fp16
__global__ void convertFloatToHalf(const float *in, size_t len, __half *out)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len)
    {
        out[tid] = fp_convert<float, __half>(in[tid]);
    }
}

template<typename T>
cudaDataType_t cublas_dtype();
template<>
cudaDataType_t cublas_dtype<float>()
{
    return CUDA_R_32F;
}
template<>
cudaDataType_t cublas_dtype<half>()
{
    return CUDA_R_16F;
}

template<typename T>
int callGruKernel(cudaStream_t stream, cublasHandle_t cublasHandle, int batch_size, int max_seq_len, int input_size, int hidden_size,
                  const T *x_weights,      // shape:[hidden_size*3, input_size], [[Wxz], [Wxr], [Wxh]]
                  const T *h_weights,      // shape:[hidden_size*3, hidden_size], [[Whz], [Whr], [Whh]]
                  const T *bias,           // shape:[hidden_size*3], [[bz], [br], [bh]]
                  T *      pre_state_cell, // shape:[batch_size, hidden_size].
                                           // This is a temp buffer for storing pre-cell hidden state
                  T *mat_wx,               // shape=[batch_size, max_seq_len, hidden_size*3].
                                           // This is a temp buffer for storing input matrix multiply results.
                                           // It represents: [[X * Wxz], [X * Wxr], [X * Wxh]]
                  T *rh_cell,              // shape=[batch_size, hidden_size].
                                           // This is a temp buffer for storing rh results
                  const T *  inputs,       // shape=[batch_size, max_seq_len, input_size]
                  const int *seq_len_arr,  // shape=[batch_size]. The sequence length of each sample.
                  T *        outputs,      // shape=[batch_size, max_seq_len, hidden_size]
                  T *        outputs_final // shape=[batch_size, hidden_size]
                                           // Only get the final state of each sample
)
{
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS_ERROR(cublasSetStream(cublasHandle, stream));

    // x[...,nDimInput] -> x[...,nDimHidden]
    CHECK_CUBLAS_ERROR(cublasGemmEx(cublasHandle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    hidden_size * 3,
                                    batch_size * max_seq_len,
                                    input_size,
                                    &alpha,
                                    x_weights,
                                    cublas_dtype<T>(),
                                    hidden_size * 3,
                                    inputs,
                                    cublas_dtype<T>(),
                                    input_size,
                                    &beta,
                                    mat_wx,
                                    cublas_dtype<T>(),
                                    hidden_size * 3,
                                    CUDA_R_32F,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // other operation
    dim3 blockSize(CEIL(hidden_size, 32), 1, 1), gridSize(1, (batch_size / blockSize.y), 1);
    gruCellPersistKernel<T><<<gridSize, blockSize, 0, stream>>>(
        batch_size,
        input_size,
        hidden_size,
        max_seq_len,
        mat_wx,
        seq_len_arr,
        h_weights,
        bias,
        pre_state_cell,
        rh_cell,
        outputs,
        outputs_final);
    return 0;
}

// Write values into buffer
template<typename T>
void writeToBuffer(char *&buffer, const T &val)
{
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
}

// Read values from buffer
template<typename T>
T readFromBuffer(const char *&buffer)
{
    T val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
    return val;
}

// ALIGNPTR
int8_t *alignPtr(int8_t *ptr, uintptr_t to)
{
    uintptr_t addr = (uintptr_t)ptr;
    if (addr % to)
    {
        addr += to - addr % to;
    }
    return (int8_t *)addr;
}

// NEXTWORKSPACEPTR
int8_t *nextWorkspacePtr(int8_t *ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t)ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t *)addr, CUDA_MEM_ALIGN);
}

GruPlugin::GruPlugin(const std::string name, const int inputSize, const int hiddenSize, float *x_weights, float *h_weights, float *bias):
    mInputSize(inputSize), mHiddenSize(hiddenSize), mLayerName(name)
{
    mWeightsX_h = (float *)malloc(mInputSize * mHiddenSize * 3 * sizeof(float));
    mWeightsH_h = (float *)malloc(mHiddenSize * mHiddenSize * 3 * sizeof(float));
    mBias_h     = (float *)malloc(mHiddenSize * 3 * sizeof(float));

    memcpy((void *)mWeightsX_h, (void *)x_weights, mInputSize * mHiddenSize * 3 * sizeof(float));
    memcpy((void *)mWeightsH_h, (void *)h_weights, mHiddenSize * mHiddenSize * 3 * sizeof(float));
    memcpy((void *)mBias_h, (void *)bias, mHiddenSize * 3 * sizeof(float));
}

GruPlugin::GruPlugin(const std::string name, const void *data, size_t length):
    mLayerName(name)
{
    const char *d = static_cast<const char *>(data), *a = d;

    mInputSize  = readFromBuffer<int>(d);
    mHiddenSize = readFromBuffer<int>(d);

    mWeightsX_h = (float *)malloc(mInputSize * mHiddenSize * 3 * sizeof(float));
    mWeightsH_h = (float *)malloc(mHiddenSize * mHiddenSize * 3 * sizeof(float));
    mBias_h     = (float *)malloc(mHiddenSize * 3 * sizeof(float));
    for (int i = 0; i < mInputSize * mHiddenSize * 3; ++i)
    {
        mWeightsX_h[i] = readFromBuffer<float>(d);
    }
    for (int i = 0; i < mHiddenSize * mHiddenSize * 3; ++i)
    {
        mWeightsH_h[i] = readFromBuffer<float>(d);
    }
    for (int i = 0; i < mHiddenSize * 3; ++i)
    {
        mBias_h[i] = readFromBuffer<float>(d);
    }
}

GruPlugin::GruPlugin(const GruPlugin &obj)
{
    mInputSize  = obj.mInputSize;
    mHiddenSize = obj.mHiddenSize;
    mWeightsX_h = (float *)malloc(mInputSize * mHiddenSize * 3 * sizeof(float));
    memcpy((void *)mWeightsX_h, (void *)obj.mWeightsX_h, mInputSize * mHiddenSize * 3 * sizeof(float));
    mWeightsH_h = (float *)malloc(mHiddenSize * mHiddenSize * 3 * sizeof(float));
    memcpy((void *)mWeightsH_h, (void *)obj.mWeightsH_h, mHiddenSize * mHiddenSize * 3 * sizeof(float));
    mBias_h = (float *)malloc(mHiddenSize * 3 * sizeof(float));
    memcpy((void *)mBias_h, (void *)obj.mBias_h, mHiddenSize * 3 * sizeof(float));
    mLayerName       = obj.mLayerName;
    mPluginNamespace = obj.mPluginNamespace;

    mWeightsX_d      = obj.mWeightsX_d;
    mWeightsH_d      = obj.mWeightsH_d;
    mBias_d          = obj.mBias_d;
    mWeightsX_half_d = obj.mWeightsX_half_d;
    mWeightsH_half_d = obj.mWeightsH_half_d;
    mBias_half_d     = obj.mBias_half_d;
    mCuBlasHandle    = obj.mCuBlasHandle;
}

GruPlugin::~GruPlugin()
{
    if (mWeightsX_h != nullptr)
    {
        free(mWeightsX_h);
        mWeightsX_h = nullptr;
    }
    if (mWeightsH_h != nullptr)
    {
        free(mWeightsH_h);
        mWeightsH_h = nullptr;
    }
    if (mBias_h != nullptr)
    {
        free(mBias_h);
        mBias_h = nullptr;
    }
}

const char *GruPlugin::getPluginType() const
{
    return GRU_PLUGIN_NAME;
}

const char *GruPlugin::getPluginVersion() const
{
    return GRU_PLUGIN_VERSION;
}

inline int GruPlugin::getNbOutputs() const
{
    return GRU_PLUGIN_NUM_OUTPUT;
}

inline int GruPlugin::initialize()
{
    CHECK_CUDA_ERROR(cudaMalloc((void **)&mWeightsX_d, mInputSize * mHiddenSize * 3 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&mWeightsH_d, mHiddenSize * mHiddenSize * 3 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&mBias_d, mHiddenSize * 3 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy((void *)mWeightsX_d, (void *)mWeightsX_h, mInputSize * mHiddenSize * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy((void *)mWeightsH_d, (void *)mWeightsH_h, mHiddenSize * mHiddenSize * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy((void *)mBias_d, (void *)mBias_h, mHiddenSize * 3 * sizeof(float), cudaMemcpyHostToDevice));
    // for fp16
    CHECK_CUDA_ERROR(cudaMalloc((void **)&mWeightsX_half_d, mInputSize * mHiddenSize * 3 * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&mWeightsH_half_d, mHiddenSize * mHiddenSize * 3 * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&mBias_half_d, mHiddenSize * 3 * sizeof(__half)));
    // convert weights precision from fp32 to fp16
    convertFloatToHalf<<<(mInputSize * mHiddenSize * 3 + 63) / 64, 64>>>(mWeightsX_d, mInputSize * mHiddenSize * 3, mWeightsX_half_d);
    convertFloatToHalf<<<(mHiddenSize * mHiddenSize * 3 + 63) / 64, 64>>>(mWeightsH_d, mHiddenSize * mHiddenSize * 3, mWeightsH_half_d);
    convertFloatToHalf<<<(mHiddenSize * 3 + 63) / 64, 64>>>(mBias_d, mHiddenSize * 3, mBias_half_d);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(0));
    CHECK_CUBLAS_ERROR(cublasCreate(&mCuBlasHandle));
    return 0;
}

inline void GruPlugin::terminate()
{
    if (mWeightsX_d != nullptr)
    {
        CHECK_CUDA_ERROR(cudaFree(mWeightsX_d));
        mWeightsX_d = nullptr;
    }
    if (mWeightsH_d != nullptr)
    {
        CHECK_CUDA_ERROR(cudaFree(mWeightsH_d));
        mWeightsH_d = nullptr;
    }
    if (mBias_d != nullptr)
    {
        CHECK_CUDA_ERROR(cudaFree(mBias_d));
        mBias_d = nullptr;
    }
    // for fp16
    if (mWeightsX_half_d != nullptr)
    {
        CHECK_CUDA_ERROR(cudaFree(mWeightsX_half_d));
        mWeightsX_half_d = nullptr;
    }
    if (mWeightsH_half_d != nullptr)
    {
        CHECK_CUDA_ERROR(cudaFree(mWeightsH_half_d));
        mWeightsH_half_d = nullptr;
    }
    if (mBias_half_d != nullptr)
    {
        CHECK_CUDA_ERROR(cudaFree(mBias_half_d));
        mBias_half_d = nullptr;
    }
    CHECK_CUBLAS_ERROR(cublasDestroy(mCuBlasHandle));
}

inline size_t GruPlugin::getSerializationSize() const
{
    size_t total_size = 0;
    total_size += sizeof(int);                                                  // mInputSize
    total_size += sizeof(int);                                                  // mHiddenSize
    total_size += mHiddenSize * 3 * (mInputSize + mHiddenSize) * sizeof(float); // mWeightsX_h and mWeightsH_h
    total_size += mHiddenSize * 3 * sizeof(float);                              // mBias_h
    return total_size;
}

inline void GruPlugin::serialize(void *buffer) const
{
    char *      d = static_cast<char *>(buffer);
    const char *a = d;
    writeToBuffer<int>(d, mInputSize);
    writeToBuffer<int>(d, mHiddenSize);
    // write weights
    for (int i = 0; i < mInputSize * mHiddenSize * 3; ++i)
    {
        writeToBuffer<float>(d, mWeightsX_h[i]);
    }
    for (int i = 0; i < mHiddenSize * mHiddenSize * 3; ++i)
    {
        writeToBuffer<float>(d, mWeightsH_h[i]);
    }
    // write bias
    for (int i = 0; i < mHiddenSize * 3; ++i)
    {
        writeToBuffer<float>(d, mBias_h[i]);
    }
}

inline void GruPlugin::destroy()
{
    delete this;
}

inline void GruPlugin::setPluginNamespace(const char *pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

inline const char *GruPlugin::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

inline DataType GruPlugin::getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

inline void GruPlugin::attachToContext(cudnnContext *, cublasContext *, IGpuAllocator *) {}

inline void GruPlugin::detachFromContext() {}

inline IPluginV2DynamicExt *GruPlugin::clone() const
{
    return new GruPlugin(*this);
}

inline DimsExprs GruPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder)
{
    DimsExprs outputDims;
    if (outputIndex == 0)
    {
        outputDims.nbDims = 3;
        outputDims.d[0]   = inputs[0].d[0];
        outputDims.d[1]   = inputs[0].d[1];
        outputDims.d[2]   = exprBuilder.constant(mHiddenSize);
    }
    else
    {
        outputDims.nbDims = 2;
        outputDims.d[0]   = inputs[0].d[0];
        outputDims.d[1]   = exprBuilder.constant(mHiddenSize);
    }
    return outputDims;
}

inline bool GruPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs)
{
    switch (pos)
    {
    case 0: // input0
        return (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF) && inOut[0].format == PluginFormat::kLINEAR;
    case 1: // input1
        return inOut[1].type == DataType::kINT32 && inOut[1].format == PluginFormat::kLINEAR;
    case 2: // output0
        return (inOut[2].type == inOut[0].type) && inOut[2].format == PluginFormat::kLINEAR;
    case 3: // output1
        return (inOut[3].type == inOut[0].type) && inOut[3].format == PluginFormat::kLINEAR;
    default:
        return false;
    }
}

inline void GruPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) {}

inline size_t GruPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const
{
    const size_t batchSize = inputs[0].dims.d[0], maxSeqLen = inputs[0].dims.d[1];
    const size_t element_size = (inputs[0].type == DataType::kFLOAT) ? sizeof(float) : sizeof(__half);
    size_t       realSize = 0, workspaceSize = 0;
    realSize = batchSize * mHiddenSize * element_size; // preStateCell_d
    workspaceSize += CEIL(realSize, CUDA_MEM_ALIGN);
    realSize = batchSize * mHiddenSize * element_size; // rhCell_d
    workspaceSize += CEIL(realSize, CUDA_MEM_ALIGN);
    realSize = batchSize * maxSeqLen * mHiddenSize * 3 * element_size; // matMulRes_d
    workspaceSize += CEIL(realSize, CUDA_MEM_ALIGN);
    return workspaceSize;
}

inline int32_t GruPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream)
{
    int      status = -1, batchSize = inputDesc[0].dims.d[0], maxSeqLen = inputDesc[0].dims.d[1];
    DataType dataType = inputDesc[0].type;

    if (dataType == DataType::kFLOAT)
    {
        auto *    preStateCell_d    = reinterpret_cast<float *>(workspace);
        uintptr_t preStateCell_size = CEIL(batchSize * mHiddenSize * sizeof(float), CUDA_MEM_ALIGN);
        auto *    rhCell_d          = reinterpret_cast<float *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(preStateCell_d), preStateCell_size));
        uintptr_t rhCell_size       = preStateCell_size;
        auto *    matMulRes_d       = reinterpret_cast<float *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(rhCell_d), rhCell_size));
        uintptr_t matMulRes_size    = CEIL(3 * batchSize * maxSeqLen * mHiddenSize * sizeof(float), CUDA_MEM_ALIGN);

        CHECK_CUDA_ERROR(cudaMemsetAsync((void *)workspace, 0, preStateCell_size + rhCell_size + matMulRes_size, stream));

        status = callGruKernel<float>(stream, mCuBlasHandle, batchSize, maxSeqLen, mInputSize, mHiddenSize, mWeightsX_d, mWeightsH_d, mBias_d, preStateCell_d, matMulRes_d, rhCell_d, (float *)inputs[0], (int *)inputs[1], (float *)outputs[0], (float *)outputs[1]);
    }
    else
    {
        auto *    preStateCell_d    = reinterpret_cast<__half *>(workspace);
        uintptr_t preStateCell_size = CEIL(batchSize * mHiddenSize * sizeof(__half), CUDA_MEM_ALIGN);

        auto *    rhCell_d    = reinterpret_cast<__half *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(preStateCell_d), preStateCell_size));
        uintptr_t rhCell_size = preStateCell_size;

        auto *    matMulRes_d    = reinterpret_cast<__half *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(rhCell_d), rhCell_size));
        uintptr_t matMulRes_size = CEIL(3 * batchSize * maxSeqLen * mHiddenSize * sizeof(__half), CUDA_MEM_ALIGN);

        CHECK_CUDA_ERROR(cudaMemsetAsync((void *)workspace, 0, preStateCell_size + rhCell_size + matMulRes_size, stream));

        status = callGruKernel<__half>(stream, mCuBlasHandle, batchSize, maxSeqLen, mInputSize, mHiddenSize, mWeightsX_half_d, mWeightsH_half_d, mBias_half_d, preStateCell_d, matMulRes_d, rhCell_d, (__half *)inputs[0], (int *)inputs[1], (__half *)outputs[0], (__half *)outputs[1]);
    }
    return status;
}

GruPluginCreator::GruPluginCreator() {}

inline const char *GruPluginCreator::getPluginName() const
{
    return GRU_PLUGIN_NAME;
}

inline const char *GruPluginCreator::getPluginVersion() const
{
    return GRU_PLUGIN_VERSION;
}

inline const PluginFieldCollection *GruPluginCreator::getFieldNames()
{
    return nullptr;
}

IPluginV2 *GruPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{
    const PluginField *fields     = fc->fields;
    int                inputSize  = *(static_cast<const int *>(fields[0].data));
    int                hiddenSize = *(static_cast<const int *>(fields[1].data));
    float *            x_weights  = const_cast<float *>(static_cast<const float *>(fields[2].data));
    float *            h_weights  = const_cast<float *>(static_cast<const float *>(fields[3].data));
    float *            bias       = const_cast<float *>(static_cast<const float *>(fields[4].data));
    return new GruPlugin(name, inputSize, hiddenSize, x_weights, h_weights, bias);
}

IPluginV2 *GruPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    return new GruPlugin(name, serialData, serialLength);
}

inline void GruPluginCreator::setPluginNamespace(const char *pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

inline const char *GruPluginCreator::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(GruPluginCreator);
