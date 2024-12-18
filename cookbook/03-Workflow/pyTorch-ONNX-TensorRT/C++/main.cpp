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

#include "calibrator.h"
#include "cnpy.h"
#include "cookbookHelper.cuh"

using namespace nvinfer1;

const std::string cookbookPath {std::getenv("TRT_COOKBOOK_PATH")};

const std::string onnxFile {cookbookPath + "/00-Data/model/model-trained.onnx"};
const std::string calibrationDataFile {cookbookPath + "/00-Data/data/CalibrationData.npy"};
const std::string inferenceDataFile {cookbookPath + "/00-Data/data/InferenceData.npy"};
const std::string trtFile {"model.trt"};
const std::string int8CacheFile {"model.Int8Cache"};
const int         nHeight {28};
const int         nWidth {28};
const Dims64      inputShape {4, {1, 1, nHeight, nWidth}};

static Logger gLogger(ILogger::Severity::kERROR);

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

        // Remove these 3 lines below to use FP32 mode
        config->setFlag(BuilderFlag::kINT8);
        MyCalibratorV1 myCalibrator(calibrationDataFile, 1, inputShape, int8CacheFile);
        config->setInt8Calibrator(&myCalibrator);

        nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser->parseFromFile(onnxFile.c_str(), int(gLogger.reportableSeverity)))
        {
            std::cout << "Fail parsing " << onnxFile << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i)
            {
                auto *error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
            }
            return;
        }
        std::cout << "Succeed parsing " << onnxFile << std::endl;

        ITensor *inputTensor = network->getInput(0);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, inputShape);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, inputShape);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims64 {4, {2, 1, nHeight, nWidth}});
        config->addOptimizationProfile(profile);

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
    context->setInputShape(tensorNameList[0], inputShape);

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

    std::map<std::string, std::tuple<void *, void *, int, DataType>> bufferMap;
    for (auto const name : tensorNameList)
    {
        Dims64   dim {context->getTensorShape(name)};
        DataType dataType     = engine->getTensorDataType(name);
        int      nByte        = std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<>()) * dataTypeToSize(dataType);
        void    *hostBuffer   = (void *)new char[nByte];
        void    *deviceBuffer = nullptr;
        CHECK(cudaMalloc(&deviceBuffer, nByte));
        bufferMap[name] = std::make_tuple(hostBuffer, deviceBuffer, nByte, dataType);
    }

    float *pInputData = static_cast<float *>(std::get<0>(bufferMap[tensorNameList[0]])); // We certainly know the data type of input tensors

    cnpy::NpyArray array_x = cnpy::npy_load(inferenceDataFile);
    memcpy(pInputData, array_x.data<float>(), std::get<2>(bufferMap[tensorNameList[0]]));

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
        void    *hostBuffer = std::get<0>(bufferMap[name]);
        DataType dataType   = std::get<3>(bufferMap[name]);
        if (dataType == DataType::kFLOAT)
        {
            printArrayInformation(static_cast<float *>(hostBuffer), name, context->getTensorShape(name), false, true);
        }
        else
        {
            printArrayInformation(static_cast<int64_t *>(hostBuffer), name, context->getTensorShape(name), false, true);
        }
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
