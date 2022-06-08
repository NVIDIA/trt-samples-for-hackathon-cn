/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace nvinfer1;

#define ck(call) check(call, __LINE__, __FILE__)

const std::string trtFile {"./model.plan"};

inline bool check(cudaError_t e, int iLine, const char *szFile)
{
    if (e != cudaSuccess)
    {
        std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << std::endl;
        return false;
    }
    return true;
}

class Logger : public ILogger
{
public:
    Severity reportableSeverity;

    Logger(Severity severity = Severity::kINFO):
        reportableSeverity(severity) {}

    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity > reportableSeverity)
        {
            return;
        }
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

static Logger gLogger(ILogger::Severity::kERROR);

void print(const float *v, Dims dimOut, std::string name)
{
    std::cout << name << ": (";
    for (int i = 0; i < dimOut.nbDims; ++i)
    {
        std::cout << dimOut.d[i] << ", ";
    }
    std::cout << "\b\b)" << std::endl;
    for (int b = 0; b < dimOut.d[0]; b++)
    {
        for (int h = 0; h < dimOut.d[1]; h++)
        {
            for (int w = 0; w < dimOut.d[2]; w++)
            {
                std::cout << std::fixed << std::setprecision(1) << std::setw(4) << v[(b * dimOut.d[0] + h) * dimOut.d[1] + w] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void run()
{
    ICudaEngine *engine = nullptr;

    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0)
        {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize, nullptr);
        if (engine == nullptr)
        {
            std::cout << "Failed building engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building engine!" << std::endl;
    }
    else
    {
        IBuilder *            builder = createInferBuilder(gLogger);
        INetworkDefinition *  network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        IBuilderConfig *      config  = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 30);

        ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims3 {-1, -1, -1});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims3 {1, 1, 1});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims3 {3, 4, 5});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims3 {6, 8, 10});
        config->addOptimizationProfile(profile);

        IIdentityLayer *identityLayer = network->addIdentity(*inputTensor);
        network->markOutput(*identityLayer->getOutput(0));
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString->size() == 0)
        {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr)
        {
            std::cout << "Failed building engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building engine!" << std::endl;

        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile)
        {
            std::cout << "Failed opening file to write" << std::endl;
            return;
        }
        if (engineString == nullptr)
        {
            std::cout << "Failed serializaing engine" << std::endl;
            return;
        }
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        if (engineFile.fail())
        {
            std::cout << "Failed saving .plan file!" << std::endl;
            return;
        }
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }

    IExecutionContext *context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims3 {3, 4, 5});

    cudaStream_t stream;
    ck(cudaStreamCreate(&stream));

    int  inputSize = 1, outputSize = 1;
    Dims inputShape  = context->getBindingDimensions(0);
    Dims outputShape = context->getBindingDimensions(1);
    for (int i = 0; i < inputShape.nbDims; ++i)
    {
        inputSize *= inputShape.d[i];
    }
    for (int i = 0; i < outputShape.nbDims; ++i)
    {
        outputSize *= outputShape.d[i];
    }
    std::vector<void *> bufferH = {nullptr, nullptr}, bufferD = {nullptr, nullptr};
    ck(cudaHostAlloc(&bufferH[0], sizeof(float) * inputSize, cudaHostAllocWriteCombined));
    ck(cudaHostAlloc(&bufferH[1], sizeof(float) * outputSize, cudaHostAllocWriteCombined));
    ck(cudaMallocAsync(&bufferD[0], sizeof(float) * inputSize, stream));
    ck(cudaMallocAsync(&bufferD[1], sizeof(float) * outputSize, stream));
    for (int i = 0; i < inputSize; ++i)
    {
        ((float *)bufferH[0])[i] = (float)i;
    }
    cudaStreamSynchronize(stream);

    ck(cudaMemcpyAsync(bufferD[0], bufferH[0], sizeof(float) * inputSize, cudaMemcpyHostToDevice, stream));
    context->enqueueV2(bufferD.data(), stream, nullptr);
    ck(cudaMemcpyAsync(bufferH[1], bufferD[1], sizeof(float) * outputSize, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    print((float *)bufferH[0], context->getBindingDimensions(0), std::string(engine->getBindingName(0)));
    print((float *)bufferH[1], context->getBindingDimensions(1), std::string(engine->getBindingName(1)));

    cudaStreamDestroy(stream);
    ck(cudaFreeHost(bufferH[0]));
    ck(cudaFreeHost(bufferH[1]));
    ck(cudaFree(bufferD[0]));
    ck(cudaFree(bufferD[1]));
    return;
}

int main()
{
    ck(cudaSetDevice(0));
    run();
    run();
    return 0;
}
