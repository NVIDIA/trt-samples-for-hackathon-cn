/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

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

#define CHECK(call) check(call, __LINE__, __FILE__)

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
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

static Logger gLogger(ILogger::Severity::kERROR);

void print(const std::vector<float> &v, int batchSize, Dims dimOut, std::string name)
{
    std::cout << name << ": (" << batchSize << ", ";
    for (int i = 0; i < dimOut.nbDims; ++i)
    {
        std::cout << dimOut.d[i] << ", ";
    }
    std::cout << "\b\b)" << std::endl;
    for (int b = 0; b < batchSize; ++b)
    {
        for (int h = 0; h < dimOut.d[0]; ++h)
        {
            for (int w = 0; w < dimOut.d[1]; ++w)
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
        runtime->destroy();
        if (engine == nullptr)
        {
            std::cout << "Failed loading engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        IBuilder *          builder = createInferBuilder(gLogger);
        INetworkDefinition *network = builder->createNetwork();
        builder->setMaxBatchSize(3);
        builder->setMaxWorkspaceSize(1 << 30);

        ITensor *       inputTensor   = network->addInput("inputT0", DataType::kFLOAT, Dims32 {2, {4, 5}});
        IIdentityLayer *identityLayer = network->addIdentity(*inputTensor);
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

        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile)
        {
            std::cout << "Failed opening file to write" << std::endl;
            return;
        }

        IHostMemory *engineString = engine->serialize();
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

    int  inputSize = 3 * 4 * 5, outputSize = 3;
    Dims outputShape = context->getBindingDimensions(1);
    for (int i = 0; i < outputShape.nbDims; ++i)
    {
        outputSize *= outputShape.d[i];
    }
    std::vector<float>  inputH0(inputSize, 1.0f), outputH0(outputSize, 0.0f);
    std::vector<void *> bufferD = {nullptr, nullptr};
    CHECK(cudaMalloc(&bufferD[0], sizeof(float) * inputSize));
    CHECK(cudaMalloc(&bufferD[1], sizeof(float) * outputSize));
    for (int i = 0; i < inputSize; ++i)
    {
        inputH0[i] = (float)i;
    }

    CHECK(cudaMemcpy(bufferD[0], inputH0.data(), sizeof(float) * inputSize, cudaMemcpyHostToDevice));
    context->execute(3, bufferD.data());
    CHECK(cudaMemcpy(outputH0.data(), bufferD[1], sizeof(float) * outputSize, cudaMemcpyDeviceToHost));

    print(inputH0, 3, context->getBindingDimensions(0), std::string(engine->getBindingName(0)));
    print(outputH0, 3, context->getBindingDimensions(1), std::string(engine->getBindingName(1)));

    context->destroy();
    engine->destroy();
    CHECK(cudaFree(bufferD[0]));
    CHECK(cudaFree(bufferD[1]));
    return;
}

int main()
{
    CHECK(cudaSetDevice(0));
    run();
    run();
    return 0;
}
