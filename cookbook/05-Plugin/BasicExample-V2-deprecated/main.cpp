/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "AddScalarPlugin.h"
#include "cookbookHelper.cuh"

using namespace nvinfer1;

const std::string trtFile {"model.trt"};
const char       *inputTensorName {"inputT0"};
Dims64            shape {3, {3, 4, 5}};
const std::string pluginFile {"./AddScalarPlugin.so"};
const std::string pluginName {"AddScalar"};
static Logger     gLogger(ILogger::Severity::kERROR);

void run()
{
    loadPluginFile(pluginFile); // Load plugin from file
    IRuntime    *runtime {createInferRuntime(gLogger)};
    ICudaEngine *engine {nullptr};

    if (access(trtFile.c_str(), F_OK) == 0)
    {
        FileStreamReader filestream(trtFile);
        engine = runtime->deserializeCudaEngine(filestream);
    }
    else
    {
        IBuilder             *builder = createInferBuilder(gLogger);
        INetworkDefinition   *network = builder->createNetworkV2(0);
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        IBuilderConfig       *config  = builder->createBuilderConfig();

        ITensor *inputTensor = network->addInput(inputTensorName, DataType::kFLOAT, Dims64 {3, {-1, -1, -1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims64 {3, {1, 1, 1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims64 {3, {3, 4, 5}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims64 {3, {6, 8, 10}});
        config->addOptimizationProfile(profile);

        float                      scalar {1.0f};
        std::vector<PluginField>   vecPF {{"scalar", &scalar, PluginFieldType::kFLOAT32, 1}};
        PluginFieldCollection      pfc {static_cast<int32_t>(vecPF.size()), vecPF.data()};
        IPluginCreator            *pluginCreator {static_cast<IPluginCreator *>(getPluginRegistry()->getCreator("AddScalar", "1", ""))};
        std::unique_ptr<IPluginV2> plugin {pluginCreator->createPlugin("AddScalar", &pfc)};

        std::vector<ITensor *> inputsVec {inputTensor};
        IPluginV2Layer        *pluginV2Layer = network->addPluginV2(inputsVec.data(), inputsVec.size(), *plugin);

        network->markOutput(*pluginV2Layer->getOutput(0));
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString == nullptr || engineString->size() == 0)
        {
            std::cout << "Fail building engine" << std::endl;
            return;
        }
        std::cout << "Succeed building engine" << std::endl;

        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile)
        {
            std::cout << "Failed opening file to write" << std::endl;
            return;
        }
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        if (engineFile.fail())
        {
            std::cout << "Fail saving engine" << std::endl;
            return;
        }
        std::cout << "Succeed saving engine (" << trtFile << ")" << std::endl;

        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
    }

    if (engine == nullptr)
    {
        std::cout << "Fail getting engine for inference" << std::endl;
        return;
    }
    std::cout << "Succeed getting engine for inference" << std::endl;

    int const                 nIO = engine->getNbIOTensors();
    std::vector<const char *> tensorNameList(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        tensorNameList[i] = engine->getIOTensorName(i);
    }

    IExecutionContext *context = engine->createExecutionContext();
    context->setInputShape(inputTensorName, shape);

    for (auto const name : tensorNameList)
    {
        TensorIOMode mode = engine->getTensorIOMode(name);
        std::cout << (mode == TensorIOMode::kINPUT ? "Input " : "Output");
        std::cout << "-> ";
        std::cout << dataTypeToString(engine->getTensorDataType(name)) << ", ";
        std::cout << shapeToString(engine->getTensorShape(name)) << ", ";
        std::cout << shapeToString(context->getTensorShape(name)) << ", ";
        std::cout << name << std::endl;
    }

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

    float *pInputData = static_cast<float *>(std::get<0>(bufferMap[inputTensorName])); // We certainly know the data type of input tensors
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
            void *hostBuffer   = std::get<0>(bufferMap[name]);
            void *deviceBuffer = std::get<1>(bufferMap[name]);
            int   nByte        = std::get<2>(bufferMap[name]);
            CHECK(cudaMemcpy(deviceBuffer, hostBuffer, nByte, cudaMemcpyHostToDevice));
        }
    }

    context->enqueueV3(0);

    for (auto const name : tensorNameList)
    {
        if (engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT)
        {
            void *hostBuffer   = std::get<0>(bufferMap[name]);
            void *deviceBuffer = std::get<1>(bufferMap[name]);
            int   nByte        = std::get<2>(bufferMap[name]);
            CHECK(cudaMemcpy(hostBuffer, deviceBuffer, nByte, cudaMemcpyDeviceToHost));
        }
    }

    for (auto const name : tensorNameList)
    {
        void *hostBuffer = std::get<0>(bufferMap[name]);
        printArrayInformation(static_cast<float *>(hostBuffer), name, context->getTensorShape(name), false, true);
    }

    for (auto const name : tensorNameList)
    {
        void *hostBuffer   = std::get<0>(bufferMap[name]);
        void *deviceBuffer = std::get<1>(bufferMap[name]);
        delete[] static_cast<char *>(hostBuffer);
        CHECK(cudaFree(deviceBuffer));
    }
    return;
}

int main()
{
    CHECK(cudaSetDevice(0));
    run();
    run();
    return 0;
}
