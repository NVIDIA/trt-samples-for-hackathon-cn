#include "MaskPlugin.h"

using namespace nvinfer1;

PluginFieldCollection MaskPluginCreator::fc_{};
std::vector<PluginField> MaskPluginCreator::attr_;

template<typename T>
__global__ void MaskKernel(int *pInput, T *pOutput0, T *pOutput1, int t4)
{
    const int nSL = pInput[blockIdx.x], bx = blockIdx.x, tx = threadIdx.x;
    const int pos = bx * t4 + tx;

    int value = int( tx * 4 >= nSL );    
    if (tx < t4)
    {
        pOutput0[pos] = T(value) * negtiveInfinity<T>();
        pOutput1[pos] = T(1) - T(value);
    }
}

int32_t MaskPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0], nThread = (inputDesc[0].dims.d[1] + 31) >> 5 << 5, t4 = (inputDesc[0].dims.d[1] >> 2) - 1;
    if (outputDesc[0].type == DataType::kFLOAT)
    {
        (MaskKernel<float>) <<<nBlock, nThread, 0, stream>>>((int*)inputs[1], (float*)outputs[0], (float*)outputs[1], t4);
    }
    else
    {
        (MaskKernel<half>)  <<<nBlock, nThread, 0, stream>>>((int*)inputs[1], (half*)outputs[0], (half*)outputs[1], t4);
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(MaskPluginCreator);

