/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
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

#include "ResourceSharePlugin.h"
#include "cookbookHelper.cuh"

#include <fstream>
#include <numeric>

using namespace nvinfer1;

std::string const trtFile {"model-resource-share.trt"};
char const       *inputTensorName {"inputT0"};
Dims64            shape {3, {2, 3, 4}};
std::string const pluginFile {"./ResourceSharePlugin.so"};
static Logger     gLogger(ILogger::Severity::kINFO);

void run()
{
    IRuntime    *runtime {createInferRuntime(gLogger)};
    ICudaEngine *engine {nullptr};

    if (access(pluginFile.c_str(), F_OK) != 0)
    {
        std::cout << "Fail finding plugin file: " << pluginFile << std::endl;
        return;
    }
    auto pluginRegistry = getPluginRegistry();
    auto handle         = pluginRegistry->loadLibrary(pluginFile.c_str());
    if (handle == nullptr)
    {
        std::cout << "Fail loading plugin library: " << pluginFile << std::endl;
        return;
    }

    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream modelFile(trtFile, std::ios::binary | std::ios::ate);
        if (!modelFile)
        {
            std::cout << "Failed opening engine file for reading" << std::endl;
            return;
        }
        std::streamsize modelSize = modelFile.tellg();
        modelFile.seekg(0, std::ios::beg);
        std::vector<char> modelData(modelSize);
        if (!modelFile.read(modelData.data(), modelSize))
        {
            std::cout << "Failed reading engine file" << std::endl;
            return;
        }
        engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());
    }
    else
    {
        IBuilder             *builder = createInferBuilder(gLogger);
        INetworkDefinition   *network = builder->createNetworkV2(0);
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        IBuilderConfig       *config  = builder->createBuilderConfig();

        ITensor *inputTensor = network->addInput(inputTensorName, DataType::kFLOAT, Dims64 {3, {-1, -1, -1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims64 {3, {1, 1, 1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, shape);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims64 {3, {6, 8, 10}});
        config->addOptimizationProfile(profile);

        int32_t const            writerSeed {2026};
        std::vector<PluginField> writerPF {{"seed", &writerSeed, PluginFieldType::kINT32, 1}};
        PluginFieldCollection    writerFC {static_cast<int32_t>(writerPF.size()), writerPF.data()};
        PluginFieldCollection    readerFC {0, nullptr};

        auto *writerCreator = static_cast<IPluginCreatorV3One *>(pluginRegistry->getCreator("ResourceWriter", "1", ""));
        auto *readerCreator = static_cast<IPluginCreatorV3One *>(pluginRegistry->getCreator("ResourceReader", "1", ""));
        if (writerCreator == nullptr || readerCreator == nullptr)
        {
            std::cout << "Fail finding plugin creators" << std::endl;
            return;
        }

        std::unique_ptr<IPluginV3> writerPlugin {writerCreator->createPlugin("ResourceWriter", &writerFC, TensorRTPhase::kBUILD)};
        std::unique_ptr<IPluginV3> readerPlugin {readerCreator->createPlugin("ResourceReader", &readerFC, TensorRTPhase::kBUILD)};
        if (writerPlugin == nullptr || readerPlugin == nullptr)
        {
            std::cout << "Fail creating plugins" << std::endl;
            return;
        }

        std::vector<ITensor *> writerInputs {inputTensor};
        IPluginV3Layer        *writerLayer = network->addPluginV3(writerInputs.data(), writerInputs.size(), nullptr, 0, *writerPlugin);
        writerLayer->setName("WriterPluginLayer");

        ITensor               *writerOut = writerLayer->getOutput(0);
        std::vector<ITensor *> readerInputs {writerOut};
        IPluginV3Layer        *readerLayer = network->addPluginV3(readerInputs.data(), readerInputs.size(), nullptr, 0, *readerPlugin);
        readerLayer->setName("ReaderPluginLayer");

        ITensor *outputTensor = readerLayer->getOutput(0);
        outputTensor->setName("outputT0");
        network->markOutput(*outputTensor);

        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString == nullptr || engineString->size() == 0)
        {
            std::cout << "Fail building engine" << std::endl;
            return;
        }

        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile)
        {
            std::cout << "Failed opening file to write" << std::endl;
            return;
        }
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());

        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
    }

    if (engine == nullptr)
    {
        std::cout << "Fail getting engine for inference" << std::endl;
        return;
    }

    int const                 nIO = engine->getNbIOTensors();
    std::vector<char const *> tensorNameList(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        tensorNameList[i] = engine->getIOTensorName(i);
    }

    IExecutionContext *context = engine->createExecutionContext();
    context->setInputShape(inputTensorName, shape);

    std::map<std::string, std::tuple<void *, void *, int>> bufferMap;
    for (auto const name : tensorNameList)
    {
        Dims64 dim {context->getTensorShape(name)};
        int    nByte        = std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<>()) * dataTypeToSize(engine->getTensorDataType(name));
        void  *hostBuffer   = (void *)new char[nByte];
        void  *deviceBuffer = nullptr;
        CHECK(cudaMalloc(&deviceBuffer, nByte));
        bufferMap[name] = std::make_tuple(hostBuffer, deviceBuffer, nByte);
    }

    float *pInputData = static_cast<float *>(std::get<0>(bufferMap[inputTensorName]));
    for (int i = 0; i < std::get<2>(bufferMap[inputTensorName]) / sizeof(float); ++i)
    {
        pInputData[i] = float(i);
    }

    for (auto const name : tensorNameList)
    {
        context->setTensorAddress(name, std::get<1>(bufferMap[name]));
    }

    for (auto const name : tensorNameList)
    {
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT)
        {
            CHECK(cudaMemcpy(std::get<1>(bufferMap[name]), std::get<0>(bufferMap[name]), std::get<2>(bufferMap[name]), cudaMemcpyHostToDevice));
        }
    }

    context->enqueueV3(0);

    for (auto const name : tensorNameList)
    {
        if (engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT)
        {
            CHECK(cudaMemcpy(std::get<0>(bufferMap[name]), std::get<1>(bufferMap[name]), std::get<2>(bufferMap[name]), cudaMemcpyDeviceToHost));
        }
    }

    for (auto const name : tensorNameList)
    {
        printArrayInformation(static_cast<float *>(std::get<0>(bufferMap[name])), name, context->getTensorShape(name), false, true);
    }

    for (auto const name : tensorNameList)
    {
        delete[] static_cast<char *>(std::get<0>(bufferMap[name]));
        CHECK(cudaFree(std::get<1>(bufferMap[name])));
    }

    pluginRegistry->deregisterLibrary(handle);
}

int main()
{
    CHECK(cudaSetDevice(0));
    run();
    run();
    return 0;
}
