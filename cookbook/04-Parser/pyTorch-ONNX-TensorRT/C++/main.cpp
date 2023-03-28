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

#include "calibrator.h"
#include "cnpy.h"
#include "cookbookHelper.cuh"

#include <NvOnnxParser.h>

using namespace nvinfer1;

const int         nHeight {28};
const int         nWidth {28};
const std::string onnxFile {"../model.onnx"};
const std::string trtFile {"./model.plan"};
const std::string dataFile {"./data.npz"};
static Logger     gLogger(ILogger::Severity::kERROR);

// for FP16 mode
const bool bFP16Mode {false};
// for INT8 mode
const bool        bINT8Mode {false};
const int         nCalibration {1};
const std::string cacheFile {"./int8.cache"};
const std::string calibrationDataFile = std::string("./data.npz");
// 使用 00-MNISTData/test 的前 100 张图片作为校正数据，并将数据保存为 .npz 供校正器使用，避免图片解码相关代码，同理推理输入数据也保存进来

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
        IBuilder *            builder     = createInferBuilder(gLogger);
        INetworkDefinition *  network     = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile *profile     = builder->createOptimizationProfile();
        IBuilderConfig *      config      = builder->createBuilderConfig();
        IInt8Calibrator *     pCalibrator = nullptr;
        if (bFP16Mode)
        {
            config->setFlag(BuilderFlag::kFP16);
        }
        if (bINT8Mode)
        {
            config->setFlag(BuilderFlag::kINT8);
            Dims32 inputShape {4, {1, 1, nHeight, nWidth}};
            pCalibrator = new MyCalibrator(calibrationDataFile, nCalibration, inputShape, cacheFile);
            if (pCalibrator == nullptr)
            {
                std::cout << std::string("Failed getting Calibrator for Int8!") << std::endl;
                return 1;
            }
            config->setInt8Calibrator(pCalibrator);
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
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 1, nHeight, nWidth}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {4, 1, nHeight, nWidth}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {8, 1, nHeight, nWidth}});
        config->addOptimizationProfile(profile);

        network->unmarkOutput(*network->getOutput(0)); // remove output tensor "y"

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

        if (bINT8Mode && pCalibrator != nullptr)
        {
            delete pCalibrator;
        }
        /*
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
        */
    }

    int                      nIO     = engine->getNbIOTensors();
    int                      nInput  = 0;
    int                      nOutput = 0;
    std::vector<std::string> vTensorName(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        vTensorName[i] = std::string(engine->getIOTensorName(i));
        nInput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
        nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
    }

    IExecutionContext *context = engine->createExecutionContext();
    context->setInputShape(vTensorName[0].c_str(), Dims32 {4, {1, 1, nHeight, nWidth}});

    for (int i = 0; i < nIO; ++i)
    {
        std::cout << std::string(i < nInput ? "Input [" : "Output[");
        std::cout << i << std::string("]-> ");
        std::cout << dataTypeToString(engine->getTensorDataType(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(engine->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(context->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << vTensorName[i] << std::endl;
    }

    std::vector<int> vTensorSize(nIO, 0);
    for (int i = 0; i < nIO; ++i)
    {
        Dims32 dim  = context->getTensorShape(vTensorName[i].c_str());
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j)
        {
            size *= dim.d[j];
        }
        vTensorSize[i] = size * dataTypeToSize(engine->getTensorDataType(vTensorName[i].c_str()));
    }

    std::vector<void *> vBufferH {nIO, nullptr};
    std::vector<void *> vBufferD {nIO, nullptr};
    for (int i = 0; i < nIO; ++i)
    {
        vBufferH[i] = (void *)new char[vTensorSize[i]];
        CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
    }

    cnpy::npz_t    npzFile = cnpy::npz_load(dataFile);
    cnpy::NpyArray array   = npzFile[std::string("inferenceData")];
    memcpy(vBufferH[0], array.data<float>(), vTensorSize[0]);

    for (int i = 0; i < nInput; ++i)
    {
        CHECK(cudaMemcpy(vBufferD[i], vBufferH[i], vTensorSize[i], cudaMemcpyHostToDevice));
    }

    for (int i = 0; i < nIO; ++i)
    {
        context->setTensorAddress(vTensorName[i].c_str(), vBufferD[i]);
    }

    context->enqueueV3(0);

    for (int i = nInput; i < nIO; ++i)
    {
        CHECK(cudaMemcpy(vBufferH[i], vBufferD[i], vTensorSize[i], cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < nInput; ++i)
    {
        printArrayInfomation((float *)vBufferH[i], context->getTensorShape(vTensorName[i].c_str()), vTensorName[i], true);
    }

    for (int i = nInput; i < nIO; ++i)
    {
        printArrayInfomation((int *)vBufferH[i], context->getTensorShape(vTensorName[i].c_str()), vTensorName[i], true, 1);
    }

    for (int i = 0; i < nIO; ++i)
    {
        delete[] vBufferH[i];
        CHECK(cudaFree(vBufferD[i]));
    }
    return 0;
}
