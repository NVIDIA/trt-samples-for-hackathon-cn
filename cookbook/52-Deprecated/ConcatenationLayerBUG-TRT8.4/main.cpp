#include "NvInfer.h"
#include "logging.h"

#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <vector>

#ifndef CUDA_CHECK
    #define CUDA_CHECK(callstr)                                                                    \
        {                                                                                          \
            cudaError_t error_code = callstr;                                                      \
            if (error_code != cudaSuccess)                                                         \
            {                                                                                      \
                std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
                assert(0);                                                                         \
            }                                                                                      \
        }
#endif // CUDA_CHECK

using namespace nvinfer1;

static Logger gLogger;

int main()
{
    int    nN = 1;
    int    nC = 1;
    int    nH = 8;
    int    nW = 8;
    float *bufferH[2];
    float *bufferD[2];
    int    nChannelList[3] = {1, 2, 4};

    IBuilder *      builder = createInferBuilder(gLogger);
    IBuilderConfig *config  = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 30);
    INetworkDefinition *  network = builder->createNetworkV2(1U);
    ITensor *             data    = network->addInput("prob0", DataType::kFLOAT, Dims4 {-1, nC, nH, nW});
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions(data->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, nC, nH, nW}});
    profile->setDimensions(data->getName(), OptProfileSelector::kOPT, Dims32 {4, {1, nC, nH, nW}});
    profile->setDimensions(data->getName(), OptProfileSelector::kMAX, Dims32 {4, {1, nC, nH, nW}});
    //profile->setDimensions(data->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, nC, nH, nW}});
    //profile->setDimensions(data->getName(), OptProfileSelector::kOPT, Dims32 {4, {2, nC, nH, nW}});
    //profile->setDimensions(data->getName(), OptProfileSelector::kMAX, Dims32 {4, {4, nC, nH, nW}});
    config->addOptimizationProfile(profile);

    float *pW = new float[4];
    float *pB = new float[4];
    for (int j = 0; j < 4; ++j)
    {
        pW[j] = 100;
        pB[j] = j;
    }

    ITensor *convList[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < 3; ++i)
    {
        auto convLayer = network->addConvolutionNd(*data,
                                                   nChannelList[i],
                                                   DimsHW {1, 1},
                                                   Weights {DataType::kFLOAT, pW, nChannelList[i]},
                                                   Weights {DataType::kFLOAT, pB, nChannelList[i]});
        convList[i]    = convLayer->getOutput(0);
    }
    auto concatLayer = network->addConcatenation(convList, 3);

    network->markOutput(*concatLayer->getOutput(0));

    ICudaEngine *      engine  = builder->buildEngineWithConfig(*network, *config);
    IExecutionContext *context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims32 {4, {nN, nC, nH, nW}});

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    //Create GPU buffers on device
    size_t nInputElement  = nN * nC * nH * nW * 1;
    size_t nOutputElement = nN * nC * nH * nW * 7;
    bufferH[0]            = reinterpret_cast<float *>(malloc(sizeof(float) * nInputElement));
    bufferH[1]            = reinterpret_cast<float *>(malloc(sizeof(float) * nOutputElement));
    CUDA_CHECK(cudaMalloc((void **)&bufferD[0], sizeof(float) * nInputElement));
    CUDA_CHECK(cudaMalloc((void **)&bufferD[1], sizeof(float) * nOutputElement));

    for (int i = 0; i < nInputElement; ++i)
    {
        bufferH[0][i] = 1;
    }
    CUDA_CHECK(cudaMemcpyAsync(bufferD[0], bufferH[0], sizeof(float) * nInputElement, cudaMemcpyHostToDevice, stream));

    context->enqueueV2((void **)bufferD, stream, nullptr);

    CUDA_CHECK(cudaMemcpyAsync(bufferH[1], bufferD[1], sizeof(float) * nOutputElement, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    for (int i = 0; i < nOutputElement; ++i)
    {
        if (i % (nH * nW) == 0)
            printf("\n#----------------nH*nW*%d\n", i / nH / nW);
        printf("%.0f,", bufferH[1][i]);
    }
    printf("\n\n");

    cudaStreamDestroy(stream);
    delete[] pW;
    delete[] pB;
    free(bufferH[0]);
    free(bufferH[1]);
    CUDA_CHECK(cudaFree(bufferD[0]));
    CUDA_CHECK(cudaFree(bufferD[1]));

    return 0;
}