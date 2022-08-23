#include "MaskPlugin.h"

using namespace nvinfer1;

PluginFieldCollection MaskPluginCreator::fc_{};
std::vector<PluginField> MaskPluginCreator::attr_;

__global__ void MaskKernel(int *pInput, int *pOutput, int t4)
{
    const int nSL = pInput[blockIdx.x], bx = blockIdx.x, tx = threadIdx.x;
    const int pos = bx * t4 + tx;
    if (tx < t4)
    {
        pOutput[pos] = int( tx * 4 >= nSL );
    }
}

int32_t MaskPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0], nThread = (inputDesc[0].dims.d[1] + 31) >> 5 << 5, t4 = (inputDesc[0].dims.d[1] >> 2) - 1;
    MaskKernel <<<nBlock, nThread, 0, stream>>>((int*)inputs[1], (int*)outputs[0], t4);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(MaskPluginCreator);

