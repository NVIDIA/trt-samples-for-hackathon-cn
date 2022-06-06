#include "LayerNormPlugin.h"
#include "layer_norm.cuh"
using namespace nvinfer1;

PluginFieldCollection    LayerNormPluginCreator::fc_ {};
std::vector<PluginField> LayerNormPluginCreator::attr_;

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

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    // #rows
    int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    // #cols
    int nValuePerBlock = inputDesc[0].dims.d[inputDesc[0].dims.nbDims-1];

    auto *    mean                  = reinterpret_cast<float *>(workspace);
    uintptr_t mean_size             = CEIL_TO(nBlock * sizeof(float), CUDA_MEM_ALIGN);
    auto *    inv_variance          = reinterpret_cast<float *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(mean), mean_size));
    uintptr_t inv_variance_size     = mean_size;
    if (inputDesc[0].type == DataType::kFLOAT && outputDesc[0].type == DataType::kFLOAT)
    {
        oneflow::cuda::layer_norm::DirectLoad<float, float> load((float *)inputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DirectStore<float, float> store((float *)outputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), float>(stream, load, store, nBlock, nValuePerBlock, (float)epsilon_, mean, inv_variance);
    }
    else if (inputDesc[0].type == DataType::kFLOAT && outputDesc[0].type == DataType::kHALF)
    {
        oneflow::cuda::layer_norm::DirectLoad<float, float> load((float *)inputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DirectStore<float, half> store((half *)outputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), float>(stream, load, store, nBlock, nValuePerBlock, (float)epsilon_, mean, inv_variance);
    }
    else if (inputDesc[0].type == DataType::kHALF && outputDesc[0].type == DataType::kFLOAT)
    {
        oneflow::cuda::layer_norm::DirectLoad<half, float> load((half *)inputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DirectStore<float, float> store((float *)outputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), float>(stream, load, store, nBlock, nValuePerBlock, (float)epsilon_, mean, inv_variance);
    }
    else if (inputDesc[0].type == DataType::kHALF && outputDesc[0].type == DataType::kHALF)
    {
        oneflow::cuda::layer_norm::DirectLoad<half, float> load((half *)inputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DirectStore<float, half> store((half *)outputs[0], nValuePerBlock);
        oneflow::cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), float>(stream, load, store, nBlock, nValuePerBlock, (float)epsilon_, mean, inv_variance);
    }
    else {
        printf("[LayerNormPlugin ERROR] Should never reach here\n");
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);
