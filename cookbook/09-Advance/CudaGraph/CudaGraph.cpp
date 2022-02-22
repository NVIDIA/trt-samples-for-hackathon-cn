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

void print(const std::vector<float> &v, Dims dimOut, std::string name)
{
    std::cout << name << ": (";
    for (int i = 0; i < dimOut.nbDims; i++)
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

ICudaEngine *loadEngine(const std::string &trtFile)
{
    std::ifstream engineFile(trtFile, std::ios::binary);
    long int      fsize = 0;

    engineFile.seekg(0, engineFile.end);
    fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    IRuntime *   runtime {createInferRuntime(gLogger)};
    ICudaEngine *engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    runtime->destroy();
    return engine;
}

ICudaEngine *loadEngine(IHostMemory *engineString)
{
    IRuntime *   runtime {createInferRuntime(gLogger)};
    ICudaEngine *engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
    runtime->destroy();
    return engine;
}

bool saveEngine(IHostMemory *engineString, const std::string &trtFile)
{
    std::ofstream engineFile(trtFile, std::ios::binary);
    if (!engineFile)
    {
        std::cout << "Failed opening file to write" << std::endl;
        return false;
    }
    if (engineString == nullptr)
    {
        std::cout << "Failed serializaing engine" << std::endl;
        return false;
    }
    engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
    return engineFile.fail();
}

void run()
{
    ICudaEngine *engine  = nullptr;
    std::string  trtFile = std::string("./engine.trt");

    if (access(trtFile.c_str(), F_OK) == 0)
    {
        engine = loadEngine(trtFile);
        if (engine == nullptr)
        {
            std::cout << "Failed loading engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        IBuilder *            builder     = createInferBuilder(gLogger);
        INetworkDefinition *  network     = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile *profile     = builder->createOptimizationProfile();
        IBuilderConfig *      config      = builder->createBuilderConfig();
        ITensor *             inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims3 {-1, -1, -1});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims3 {1, 1, 1});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims3 {3, 4, 5});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims3 {6, 8, 10});
        config->addOptimizationProfile(profile);

        IIdentityLayer *identityLayer = network->addIdentity(*inputTensor);
        network->markOutput(*identityLayer->getOutput(0));
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        network->destroy();
        builder->destroy();
        if (engineString == nullptr)
        {
            std::cout << "Failed building engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building engine!" << std::endl;
        saveEngine(engineString, trtFile);
        engine = loadEngine(engineString);
    }

    IExecutionContext *context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims3 {3, 4, 5});

    cudaStream_t stream;
    ck(cudaStreamCreate(&stream));

    int  inputSize = 3 * 4 * 5, outputSize = 1;
    Dims outputShape = context->getBindingDimensions(1);
    for (int i = 0; i < outputShape.nbDims; i++)
    {
        outputSize *= outputShape.d[i];
    }
    std::vector<float>  inputH0(inputSize, 1.0f);
    std::vector<float>  outputH0(outputSize, 0.0f);
    std::vector<void *> binding = {nullptr, nullptr};
    ck(cudaMalloc(&binding[0], sizeof(float) * inputSize));
    ck(cudaMalloc(&binding[1], sizeof(float) * outputSize));
    for (int i = 0; i < inputSize; i++)
    {
        inputH0[i] = (float)i;
    }

    // 首次捕获 CUDA Graph 并运行
    cudaGraph_t     graph;
    cudaGraphExec_t graphExec = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    ck(cudaMemcpyAsync(binding[0], inputH0.data(), sizeof(float) * inputSize, cudaMemcpyHostToDevice, stream));
    context->enqueueV2(binding.data(), stream, nullptr);
    ck(cudaMemcpyAsync(outputH0.data(), binding[1], sizeof(float) * outputSize, cudaMemcpyDeviceToHost, stream));
    //cudaStreamSynchronize(stream);                        // 不用在 graph 内同步
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    print(outputH0, context->getBindingDimensions(1), std::string("outputH0Big"));

    // 输入尺寸改变后，需要首先运行一次推理，然后重新捕获 CUDA Graph，最后再运行
    context->setBindingDimensions(0, Dims3 {2, 3, 4});

    inputSize   = 2 * 3 * 4;
    outputSize  = 1;
    outputShape = context->getBindingDimensions(1);
    for (int i = 0; i < outputShape.nbDims; i++)
        outputSize *= outputShape.d[i];
    inputH0  = std::vector<float>(inputSize, 1.0f);
    outputH0 = std::vector<float>(outputSize, 0.0f);
    for (int i = 0; i < inputSize; i++)
        inputH0[i] = -(float)i;

    ck(cudaMemcpyAsync(binding[0], inputH0.data(), sizeof(float) * inputSize, cudaMemcpyHostToDevice, stream));
    context->enqueueV2(binding.data(), stream, nullptr);
    ck(cudaMemcpyAsync(outputH0.data(), binding[1], sizeof(float) * outputSize, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    ck(cudaMemcpyAsync(binding[0], inputH0.data(), sizeof(float) * inputSize, cudaMemcpyHostToDevice, stream));
    context->enqueueV2(binding.data(), stream, nullptr);
    ck(cudaMemcpyAsync(outputH0.data(), binding[1], sizeof(float) * outputSize, cudaMemcpyDeviceToHost, stream));
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    print(outputH0, context->getBindingDimensions(1), std::string("outputH0Small"));

    context->destroy();
    engine->destroy();
    cudaStreamDestroy(stream);
    ck(cudaFree(binding[0]));
    ck(cudaFree(binding[1]));
    return;
}

int main()
{
    ck(cudaSetDevice(0));
    run();
    run();
    return 0;
}
