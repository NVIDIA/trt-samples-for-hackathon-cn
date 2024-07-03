/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

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

#include "cookbookHelper.cuh"

using namespace nvinfer1;

const std::string trtFile {"model.trt"};
const char       *inputTensorName {"inputT0"};
Dims64            shape {3, {3, 4, 5}};
static Logger     gLogger(ILogger::Severity::kERROR);

void run()
{
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

        IIdentityLayer *identityLayer = network->addIdentity(*inputTensor);
        network->markOutput(*identityLayer->getOutput(0));
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        engine                    = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
    }

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
        int    nByte  = std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<>()) * dataTypeToSize(engine->getTensorDataType(name));
        void  *buffer = nullptr;
        CHECK(cudaHostAlloc(&buffer, nByte, cudaHostAllocWriteCombined));
        bufferMap[name] = std::make_tuple(nullptr, buffer, nByte);
    }

    float *pInputData = static_cast<float *>(std::get<1>(bufferMap[inputTensorName])); // We certainly know the data type of input tensors
    for (int i = 0; i < std::get<2>(bufferMap[inputTensorName]) / sizeof(float); ++i)
    {
        pInputData[i] = float(i);
    }

    for (auto const name : tensorNameList)
    {
        context->setTensorAddress(name, std::get<1>(bufferMap[name]));
    }

    context->enqueueV3(0);

    for (auto const name : tensorNameList)
    {
        void *buffer = std::get<1>(bufferMap[name]);
        printArrayInformation(static_cast<float *>(buffer), name, context->getTensorShape(name), false, true);
    }

    for (auto const name : tensorNameList)
    {
        CHECK(cudaFreeHost(std::get<1>(bufferMap[name])));
    }
    return;
}

int main()
{
    CHECK(cudaSetDevice(0));
    run();
    return 0;
}
