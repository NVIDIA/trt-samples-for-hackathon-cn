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

    void log(Severity severity, const char *msg) override
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

void print(const std::vector<float> &v, int batchSize, Dims dimOut, std::string name)
{
    std::cout << name << ": (" << batchSize << ", ";
    for (int i = 0; i < dimOut.nbDims; i++)
    {
        std::cout << dimOut.d[i] << ", ";
    }
    std::cout << "\b\b)" << std::endl;
    for (int b = 0; b < batchSize; b++)
    {
        for (int h = 0; h < dimOut.d[0]; h++)
        {
            for (int w = 0; w < dimOut.d[1]; w++)
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

bool saveEngine(const ICudaEngine *engine, const std::string &trtFile)
{
    std::ofstream engineFile(trtFile, std::ios::binary);
    if (!engineFile)
    {
        std::cout << "Failed opening file to write" << std::endl;
        return false;
    }
    IHostMemory *serializedEngine {engine->serialize()};
    if (serializedEngine == nullptr)
    {
        std::cout << "Failed serializaing engine" << std::endl;
        return false;
    }
    engineFile.write(static_cast<char *>(serializedEngine->data()), serializedEngine->size());
    serializedEngine->destroy();
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
        IBuilder *builder = createInferBuilder(gLogger);
        builder->setMaxBatchSize(3);
        INetworkDefinition *network       = builder->createNetwork();
        ITensor *           inputTensor   = network->addInput("inputT0", DataType::kFLOAT, Dims2 {4, 5});
        IIdentityLayer *    identityLayer = network->addIdentity(*inputTensor);

        network->markOutput(*identityLayer->getOutput(0));
        engine = builder->buildCudaEngine(*network);
        network->destroy();
        builder->destroy();
        if (engine == nullptr)
        {
            std::cout << "Failed building engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building engine!" << std::endl;
        saveEngine(engine, trtFile);
    }

    IExecutionContext *context = engine->createExecutionContext();

    cudaStream_t stream;
    ck(cudaStreamCreate(&stream));

    int  inputSize = 3 * 4 * 5, outputSize = 3;
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

    ck(cudaMemcpyAsync(binding[0], inputH0.data(), sizeof(float) * inputSize, cudaMemcpyHostToDevice, stream));
    context->enqueue(3, binding.data(), stream, nullptr);
    ck(cudaMemcpyAsync(outputH0.data(), binding[1], sizeof(float) * outputSize, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    print(inputH0, 3, context->getBindingDimensions(0), std::string("inputH0"));
    print(outputH0, 3, context->getBindingDimensions(1), std::string("outputH0"));

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
