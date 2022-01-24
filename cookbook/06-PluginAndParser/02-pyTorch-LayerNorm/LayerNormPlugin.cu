#include "LayerNormPlugin.h"

using namespace nvinfer1;
using namespace plugin;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

template<int n>
__global__ void layerNormKernel(T *pInput, T *pGamma, T *pBeta, T *pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * n + tx;
    T _x = pInput[index] * prevScale, _b = pGamma[tx], _a = pBeta[tx];

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

    pOutput[index] = ((moment) * (T)rsqrtf(var_shared + T(EPSILON)) * _b + _a) * postScale;
}

template <typename T>
int32_t LayerNormPlugin<T>::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0]; nValuePerBlock = inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
    switch(nValuePerBlock)
    {
    case 60:    // 仅演示 cIn*hIn*wIn==60 的情况
        (layerNormKernel<60>) <<<nBlock, nValuePerBlock, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
        break;
    default:    // shoulf NOT be here
        printf("[LayerNormPlugin<T>::enqueue] nValuePerBlock is not in [320,560,640]\n");
        break;
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

