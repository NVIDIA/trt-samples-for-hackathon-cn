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

const std::string weightFile {cookbookPath + "/00-Data/model/model-trained.npz"};
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

        ITensor *inputTensor = network->addInput("x", DataType::kFLOAT, Dims64 {4, {-1, 1, nHeight, nWidth}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, inputShape);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, inputShape);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims64 {4, {2, 1, nHeight, nWidth}});
        config->addOptimizationProfile(profile);

        cnpy::npz_t    npzFile = cnpy::npz_load(weightFile);
        cnpy::NpyArray array;
        Weights        w;
        Weights        b;
        array    = npzFile[std::string("conv1.weight")];
        w        = Weights {DataType::kFLOAT, array.data<float>(), array.num_bytes() / sizeof(float)};
        array    = npzFile[std::string("conv1.bias")];
        b        = Weights {DataType::kFLOAT, array.data<float>(), array.num_bytes() / sizeof(float)};
        auto *_0 = network->addConvolutionNd(*inputTensor, 32, DimsHW {5, 5}, w, b);
        _0->setPaddingNd(DimsHW {2, 2});
        auto *_1 = network->addActivation(*_0->getOutput(0), ActivationType::kRELU);
        auto *_2 = network->addPoolingNd(*_1->getOutput(0), PoolingType::kMAX, DimsHW {2, 2});
        _2->setStrideNd(DimsHW {2, 2});

        array    = npzFile[std::string("conv2.weight")];
        w        = Weights {DataType::kFLOAT, array.data<float>(), array.num_bytes() / sizeof(float)};
        array    = npzFile[std::string("conv2.bias")];
        b        = Weights {DataType::kFLOAT, array.data<float>(), array.num_bytes() / sizeof(float)};
        auto *_3 = network->addConvolutionNd(*_2->getOutput(0), 64, DimsHW {5, 5}, w, b);
        _3->setPaddingNd(DimsHW {2, 2});
        auto *_4 = network->addActivation(*_3->getOutput(0), ActivationType::kRELU);
        auto *_5 = network->addPoolingNd(*_4->getOutput(0), PoolingType::kMAX, DimsHW {2, 2});
        _5->setStrideNd(DimsHW {2, 2});

        auto *_6 = network->addShuffle(*_5->getOutput(0));
        _6->setReshapeDimensions(Dims64 {2, {-1, 64 * 7 * 7}});

        array     = npzFile[std::string("gemm1.weight")];
        w         = Weights {DataType::kFLOAT, array.data<float>(), array.num_bytes() / sizeof(float)};
        array     = npzFile[std::string("gemm1.bias")];
        b         = Weights {DataType::kFLOAT, array.data<float>(), array.num_bytes() / sizeof(float)};
        auto *_7  = network->addConstant(Dims64 {2, {1024, 64 * 7 * 7}}, w);
        auto *_8  = network->addMatrixMultiply(*_6->getOutput(0), MatrixOperation::kNONE, *_7->getOutput(0), MatrixOperation::kTRANSPOSE);
        auto *_9  = network->addConstant(Dims64 {2, {1, 1024}}, b);
        auto *_10 = network->addElementWise(*_8->getOutput(0), *_9->getOutput(0), ElementWiseOperation::kSUM);
        auto *_11 = network->addActivation(*_10->getOutput(0), ActivationType::kRELU);

        array     = npzFile[std::string("gemm2.weight")];
        w         = Weights {DataType::kFLOAT, array.data<float>(), array.num_bytes() / sizeof(float)};
        array     = npzFile[std::string("gemm2.bias")];
        b         = Weights {DataType::kFLOAT, array.data<float>(), array.num_bytes() / sizeof(float)};
        auto *_12 = network->addConstant(Dims64 {2, {10, 1024}}, w);
        auto *_13 = network->addMatrixMultiply(*_11->getOutput(0), MatrixOperation::kNONE, *_12->getOutput(0), MatrixOperation::kTRANSPOSE);
        auto *_14 = network->addConstant(Dims64 {2, {1, 10}}, b);
        auto *_15 = network->addElementWise(*_13->getOutput(0), *_14->getOutput(0), ElementWiseOperation::kSUM);

        auto *_16 = network->addSoftMax(*_15->getOutput(0));
        _16->setAxes(1U << 1);

        auto *_17 = network->addTopK(*_16->getOutput(0), TopKOperation::kMAX, 1, 1U << 1);

        _16->getOutput(0)->setName("y");
        _17->getOutput(1)->setName("z");

        network->markOutput(*_16->getOutput(0));
        network->markOutput(*_17->getOutput(1));

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
            printArrayInformation(static_cast<int32_t *>(hostBuffer), name, context->getTensorShape(name), false, true);
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
