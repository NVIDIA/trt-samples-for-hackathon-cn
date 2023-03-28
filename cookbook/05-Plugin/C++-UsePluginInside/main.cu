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

#include "AddScalarPlugin.h"
#include "cnpy.h"
#include "cookbookHelper.cuh"

#include <NvOnnxParser.h>

using namespace nvinfer1;

const std::string onnxFile {"./model.onnx"};
const std::string trtFile {"./model.plan"};
const std::string dataFile {"./data.npz"};
static Logger     gLogger(ILogger::Severity::kERROR);

// for FP16 mode
const bool bFP16Mode {false};

int main()
{
    CHECK(cudaSetDevice(0));
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
            return 1;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr)
        {
            std::cout << "Failed loading engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        IBuilder *            builder = createInferBuilder(gLogger);
        INetworkDefinition *  network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        IBuilderConfig *      config  = builder->createBuilderConfig();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 20 << 30);
        if (bFP16Mode)
        {
            config->setFlag(BuilderFlag::kFP16);
        }

        nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser->parseFromFile(onnxFile.c_str(), int(gLogger.reportableSeverity)))
        {
            std::cout << std::string("Failed parsing .onnx file!") << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i)
            {
                auto *error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
            }

            return 1;
        }
        std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

        ITensor *inputTensor = network->getInput(0);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 1, 64, 64}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {4, 1, 64, 64}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {8, 1, 64, 64}});
        config->addOptimizationProfile(profile);

        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString == nullptr || engineString->size() == 0)
        {
            std::cout << "Failed building serialized engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded building serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr)
        {
            std::cout << "Failed building engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded building engine!" << std::endl;

        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile)
        {
            std::cout << "Failed opening file to write" << std::endl;
            return 1;
        }
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        if (engineFile.fail())
        {
            std::cout << "Failed saving .plan file!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }

    IExecutionContext *context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims32 {4, {4, 1, 64, 64}});
    std::cout << std::string("Binding all? ") << std::string(context->allInputDimensionsSpecified() ? "Yes" : "No") << std::endl;
    int nBinding = engine->getNbBindings();
    int nInput   = 0;
    for (int i = 0; i < nBinding; ++i)
    {
        nInput += int(engine->bindingIsInput(i));
    }
    //int nOutput = nBinding - nInput;
    for (int i = 0; i < nBinding; ++i)
    {
        std::cout << std::string("Bind[") << i << std::string(i < nInput ? "]:i[" : "]:o[") << (i < nInput ? i : i - nInput) << std::string("]->");
        std::cout << dataTypeToString(engine->getBindingDataType(i)) << std::string(" ");
        std::cout << shapeToString(context->getBindingDimensions(i)) << std::string(" ");
        std::cout << engine->getBindingName(i) << std::endl;
    }

    std::vector<int> vBindingSize(nBinding, 0);
    for (int i = 0; i < nBinding; ++i)
    {
        Dims32 dim  = context->getBindingDimensions(i);
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j)
        {
            size *= dim.d[j];
        }
        vBindingSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
    }

    std::vector<void *> vBufferH {nBinding, nullptr};
    std::vector<void *> vBufferD {nBinding, nullptr};
    for (int i = 0; i < nBinding; ++i)
    {
        vBufferH[i] = (void *)new char[vBindingSize[i]];
        CHECK(cudaMalloc(&vBufferD[i], vBindingSize[i]));
    }

    cnpy::npz_t    npzFile = cnpy::npz_load(dataFile);
    cnpy::NpyArray array   = npzFile[std::string("inferenceData")];
    memcpy(vBufferH[0], array.data<float>(), vBindingSize[0]);

    for (int i = 0; i < nInput; ++i)
    {
        CHECK(cudaMemcpy(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice));
    }

    context->executeV2(vBufferD.data());

    for (int i = nInput; i < nBinding; ++i)
    {
        CHECK(cudaMemcpy(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost));
    }

    printArrayInfomation((float *)vBufferH[0], context->getBindingDimensions(0), std::string(engine->getBindingName(0)));
    printArrayInfomation((float *)vBufferH[1], context->getBindingDimensions(1), std::string(engine->getBindingName(1)));

    for (int i = 0; i < nBinding; ++i)
    {
        CHECK(cudaFree(vBufferD[i]));
    }

    return 0;
}
