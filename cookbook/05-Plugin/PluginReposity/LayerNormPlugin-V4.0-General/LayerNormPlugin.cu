#include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

/*
template<typename T>
__global__ void layerNormKernel(T *pInput, float *pGamma, float *pBeta, T *pOutput)
{
    const int n = 256;
    const int tx = threadIdx.x, index = blockIdx.x * n + tx;
    T _x = pInput[index], _b = (T)pGamma[tx], _a = (T)pBeta[tx];

    __shared__ T mean_shared, var_shared;

    typedef cub::BlockReduce<T, n> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    T& ref0 = _x;
    T sum = BlockReduce(temp).Sum(ref0);
    //__syncthreads();
    if(tx == 0)
        mean_shared = sum / T(n);
    __syncthreads();

    T moment = _x - mean_shared, moment2 = moment * moment;
    T& ref1 = moment2;
    T var = BlockReduce(temp).Sum(ref1);
    //__syncthreads();
    if(tx == 0)
        var_shared = var / T(n);
    __syncthreads();

    pOutput[index] = (moment) * (T)rsqrtf(var_shared + epsilon<T>()) * _b + _a;
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    const int nValuePerBlock = 256;    
    //if (inputDesc[0].type == DataType::kFLOAT)
    if(true)
    {
        (layerNormKernel<float>) <<<nBlock, nValuePerBlock, 0, stream>>>((float *)inputs[0], (float *)inputs[1], (float *)inputs[2], (float *)outputs[0]);
    }
    else
    {
        (layerNormKernel<half>) <<<nBlock, nValuePerBlock, 0, stream>>>((half *)inputs[0], (float *)inputs[1], (float *)inputs[2], (half *)outputs[0]);
    }
    return 0;
}
*/

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

struct mySum
{
    __host__ __device__ __forceinline__ float2 operator()(const float2 &a, const float2 &b) const
    {
        return make_float2(a.x + b.x, a.y + b.y);
    }
};

template <typename T, int TPB, int VPT>
__global__ void layerNormKernel(const T* input, const T* gamma, const T* beta, T* output)
{
    const int idx = blockIdx.x * 256 + threadIdx.x * VPT;
    T localX[VPT], localGamma[VPT], localBeta[VPT];

    copy<sizeof(T) * VPT>(&input[idx], localX);
    float2 localFloat2 = {0.f,0.f};

    const float rld = float(1)/ float(256);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const float tmp = rld * (float)localX[it];
        localFloat2.x += tmp;
        localFloat2.y += tmp * (float)localX[it];
    }

    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], localBeta);
    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], localGamma);

    using BlockReduce = cub::BlockReduce<float2, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float mu;     // mean
    __shared__ float rsigma; // 1 / std.dev.

    //const float2 sumKV = BlockReduce(temp_storage).Reduce(localFloat2, cub::Sum());
    const float2 sumKV = BlockReduce(temp_storage).Reduce(localFloat2, mySum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.x;
        rsigma = rsqrt(sumKV.y - mu * mu + 1e-6);
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        localX[it] = (float)localGamma[it] * ((float)localX[it] - mu) * rsigma + (float)localBeta[it];
    }

    copy<sizeof(T) * VPT>(localX, &output[idx]);
}

template __global__ void layerNormKernel<float, 64, 4>(const float*, const float*, const float*, float*);
template __global__ void layerNormKernel<half, 32, 8>(const half*, const half*, const half*, half*);

int LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int gridSize = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];

    if (inputDesc[0].type == DataType::kFLOAT)
    {
        constexpr int VPT = 16 / sizeof(float);
        constexpr int TPB = 256 / VPT;
        (layerNormKernel<float, TPB, VPT>)   <<<gridSize, TPB, 0, stream>>>  ((const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (float*)outputs[0]);
    }
    else
    {
        constexpr int VPT = 16 / sizeof(half);
        constexpr int TPB = 256 / VPT;
        (layerNormKernel<half, TPB, VPT>)    <<<gridSize, TPB, 0, stream>>>  ((const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (half*)outputs[0]);
    }
    return 0;
}

/*
template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

template <typename T>
using kvp = cub::KeyValuePair<T, T>;

struct mySum
{
    __host__ __device__ __forceinline__ kvp<float> operator()(const kvp<float> &a, const kvp<float> &b) const
    {
        return kvp<float>(a.key + b.key, a.value + b.value);
    }
};

template <typename OP_T, typename T, int TPB, int VPT>
__global__ void layerNormKernel(const int ld, const T* input, T* output, const T* beta, const T* gamma)
{
    const int idx = ld * blockIdx.x + threadIdx.x * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    T in_local[VPT];
    T skip_local[VPT];
    T bias_local[VPT];

    copy<sizeof(T) * VPT>(&input[idx], in_local);
    OP_T local = 0.f;
    OP_T local2 = 0.f;

    const OP_T rld = OP_T(1) / OP_T(ld);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const OP_T tmp = rld * (OP_T)in_local[it];
        local += tmp;
        local2 += tmp * (OP_T)in_local[it];
    }

    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], bias_local);
    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], skip_local);

    using BlockReduce = cub::BlockReduce<kvp<OP_T>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    using DT = OP_T;
    __shared__ DT mu;     // mean
    __shared__ DT rsigma; // 1 / std.dev.

    //const auto sumKV = BlockReduce(temp_storage).Reduce(kvp<OP_T>(local, local2), cub::Sum());
    const auto sumKV = BlockReduce(temp_storage).Reduce(kvp<OP_T>(local, local2), mySum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + 1e-6);
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        in_local[it] = (OP_T)skip_local[it] * ((OP_T)in_local[it] - mu) * rsigma + (OP_T)bias_local[it];
    }

    copy<sizeof(T) * VPT>(in_local, &output[idx]);
}

template __global__ void layerNormKernel<float, float, 64, 4>(const int ld, const float* input, float* output, const float* beta, const float* gamma);
template __global__ void layerNormKernel<float, half, 32, 8>(const int ld, const half* input, half* output, const half* beta, const half* gamma);

int LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int gridSize = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];

    if (inputDesc[0].type == DataType::kFLOAT)
    {
        constexpr int VPT = 16 / sizeof(float);
        constexpr int TPB = 256 / VPT;
        (layerNormKernel<float, float, TPB, VPT>)   <<<gridSize, TPB, 0, stream>>>  (256, (float*)inputs[0], (float*)outputs[0], (float*)inputs[2], (float*)inputs[1]);
    }
    else
    {
        constexpr int VPT = 16 / sizeof(half);
        constexpr int TPB = 256 / VPT;
        (layerNormKernel<float, half, TPB, VPT>)    <<<gridSize, TPB, 0, stream>>>  (256, (half*)inputs[0], (half*)outputs[0], (half*)inputs[2], (half*)inputs[1]);
    }
    return 0;
}
*/
REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

