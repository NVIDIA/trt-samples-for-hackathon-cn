#include "cuda_fp16.h"
#include <cuda_runtime.h>
#include "test.h"

int LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    int output[8];
    for(int i=0;i<inputDesc->dims.nbDims;i++)
        output[i] = inputDesc->dims.d[i];
    cudaMemcpyAsync((int*)outputs[0],output,sizeof(int)*inputDesc->dims.nbDims,cudaMemcpyHostToDevice,stream);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(AddPluginCreator);

