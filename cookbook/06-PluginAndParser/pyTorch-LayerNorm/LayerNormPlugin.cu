#include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

template<typename T, int n>
__global__ void layerNormKernel(T *pInput, T *pOutput, float epsilon)
{
    const int tx = threadIdx.x, index = blockIdx.x * n + threadIdx.x;

    T _x = pInput[index];

    __shared__ T mean_shared, var_shared;

    typedef cub::BlockReduce<T, n> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    T& ref0 = _x;
    T sum = BlockReduce(temp).Sum(ref0);
    //__syncthreads();
    if(tx == 0)
        mean_shared = sum / (T)n;
    __syncthreads();

    T moment = _x - mean_shared, moment2 = moment * moment;
    T& ref1 = moment2;
    T var = BlockReduce(temp).Sum(ref1);
    //__syncthreads();
    if(tx == 0)
        var_shared = var / (T)n;
    __syncthreads();

    pOutput[index] = moment * (T)rsqrtf(var_shared + (T)epsilon);
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nBlock = inputDesc[0].dims.d[0], nValuePerBlock = 1;
    for (int i=1;i< inputDesc[0].dims.nbDims;++i)
    {
        nValuePerBlock *= inputDesc[0].dims.d[i];
    }

    if(inputDesc[0].type == DataType::kFLOAT)
    {
        switch(nValuePerBlock)
        {
        case 60:    // 仅演示 cIn*hIn*wIn==60 的情况
            (layerNormKernel<float,60>) <<<nBlock, nValuePerBlock, 0, stream>>>((float *)inputs[0], (float *)outputs[0], epsilon_);
            break;
        default:    // shoulf NOT be here
            printf("[LayerNormPlugin::enqueue] nValuePerBlock = %d is not supported\n",nValuePerBlock);
            break;
        }
    }
    else
    {
        switch(nValuePerBlock)
        {
        case 60:    // 仅演示 cIn*hIn*wIn==60 的情况
            (layerNormKernel<half,60>) <<<nBlock, nValuePerBlock, 0, stream>>>((half *)inputs[0], (half *)outputs[0], epsilon_);
            break;
        default:    // shoulf NOT be here
            printf("[LayerNormPlugin::enqueue] nValuePerBlock = %d is not supported\n",nValuePerBlock);
            break;
        }
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

