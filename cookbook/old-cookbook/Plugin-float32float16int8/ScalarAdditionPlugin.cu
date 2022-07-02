// ScalarAdditionPlugin.cu
#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include "ScalarAdditionPlugin.h"

template<typename T>
__global__ void addValue(T *pDst, T *pSrc, int n, T addend)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n)
        return;
    pDst[x] = pSrc[x] + addend;
}

int ScalarAdditionPlugin::enqueue(int nBatch, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream)
{
    int n = nBatch;
    for (int i = 0; i < m.inputDim.nbDims; i++)
        n *= m.inputDim.d[i];

    //printf("(addend,scale,isINT8)=(%f,%f,%d)\n",m.addend,m.scale,m.dataType==nvinfer1::DataType::kINT8);
    switch (m.dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        {
            //printf("launch float32 kernel\n");
            addValue<<<CEIL(n,1024), 1024>>>((float *)outputs[0], (float *)inputs[0], n, (float)m.addend);
            break;
        }
    case nvinfer1::DataType::kHALF:
        {
            //printf("launch float16 kernel\n");
            addValue<<<CEIL(n,1024), 1024>>>((__half *)outputs[0], (__half *)inputs[0], n, (__half)m.addend);
            break;
        }
    case nvinfer1::DataType::kINT8:
        {
            //printf("launch int8 kernel\n");
            addValue<<<CEIL(n,1024), 1024>>>((int8_t *)outputs[0], (int8_t *)inputs[0], n, (int8_t)(m.addend / m.scale));
            break;
        }
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(ScalarAdditionPluginCreator);

