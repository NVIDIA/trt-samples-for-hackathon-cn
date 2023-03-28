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
#include "cookbookHelper.cuh"

using namespace nvinfer1;

const int         nH {28};
const int         nW {28};
const std::string trtFile {"./model.plan"};
static Logger     gLogger(ILogger::Severity::kERROR);

const int         nCalibration {10};
const int         nBatchSize {1};
const std::string cacheFile {"model.INT8Cache"};

int main()
{
    CHECK(cudaSetDevice(0));
    IBuilder *            builder     = createInferBuilder(gLogger);
    INetworkDefinition *  network     = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    IOptimizationProfile *profile     = builder->createOptimizationProfile();
    IBuilderConfig *      config      = builder->createBuilderConfig();
    IInt8Calibrator *     pCalibrator = nullptr;

    std::map<std::string, Dims32>   shapeMap;
    std::map<std::string, DataType> dataTypeMap;
    shapeMap[std::string("inputT0")]    = Dims32 {4, {1, 1, nH, nW}};
    dataTypeMap[std::string("inputT0")] = DataType::kFLOAT;

    pCalibrator = new MyCalibrator(nCalibration, nBatchSize, shapeMap, dataTypeMap, cacheFile);
    if (pCalibrator == nullptr)
    {
        std::cout << std::string("Failed getting Calibrator for Int8!") << std::endl;
        return 1;
    }
    config->setFlag(BuilderFlag::kINT8);
    config->setInt8Calibrator(pCalibrator);

    ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims32 {4, {-1, 1, nH, nW}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 1, nH, nW}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {4, 1, nH, nW}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {8, 1, nH, nW}});
    config->addOptimizationProfile(profile);

    Weights w;
    Weights b;
    float * pWeight = new float[64 * 7 * 7 * 1024]; // fake weights
    //memset(pWeight, 0, sizeof(float) * 64 * 7 * 7 * 1024);   // use all 0 weights wiil raise error below in MyCalibrator::writeCalibrationCache
    // ERROR: 2: Assertion getter(i) != 0 failed.
    // ERROR: 2: [weightConvertors.cpp::quantizeBiasCommon::310] Error Code 2: Internal Error (Assertion getter(i) != 0 failed. )
    srand(31193);
    for (int i = 0; i < 64 * 7 * 7 * 1024; ++i)
    {
        pWeight[i] = float(rand()) / RAND_MAX * 2 - 1;
    }

    w        = Weights {DataType::kFLOAT, pWeight, 32 * 1 * 5 * 5};
    b        = Weights {DataType::kFLOAT, pWeight, 32};
    auto *_0 = network->addConvolutionNd(*inputTensor, 32, DimsHW {5, 5}, w, b);
    _0->setPaddingNd(DimsHW {2, 2});
    auto *_1 = network->addActivation(*_0->getOutput(0), ActivationType::kRELU);
    auto *_2 = network->addPoolingNd(*_1->getOutput(0), PoolingType::kMAX, DimsHW {2, 2});
    _2->setStrideNd(DimsHW {2, 2});

    w        = Weights {DataType::kFLOAT, pWeight, 64 * 32 * 5 * 5};
    b        = Weights {DataType::kFLOAT, pWeight, 64};
    auto *_3 = network->addConvolutionNd(*_2->getOutput(0), 64, DimsHW {5, 5}, w, b);
    _3->setPaddingNd(DimsHW {2, 2});
    auto *_4 = network->addActivation(*_3->getOutput(0), ActivationType::kRELU);
    auto *_5 = network->addPoolingNd(*_4->getOutput(0), PoolingType::kMAX, DimsHW {2, 2});
    _5->setStrideNd(DimsHW {2, 2});

    auto *_6 = network->addShuffle(*_5->getOutput(0));
    _6->setReshapeDimensions(Dims32 {2, {-1, 64 * 7 * 7}});

    w         = Weights {DataType::kFLOAT, pWeight, 64 * 7 * 7 * 1024};
    b         = Weights {DataType::kFLOAT, pWeight, 1024};
    auto *_7  = network->addConstant(Dims32 {2, {1024, 64 * 7 * 7}}, w);
    auto *_8  = network->addMatrixMultiply(*_6->getOutput(0), MatrixOperation::kNONE, *_7->getOutput(0), MatrixOperation::kTRANSPOSE);
    auto *_9  = network->addConstant(Dims32 {2, {1, 1024}}, b);
    auto *_10 = network->addElementWise(*_8->getOutput(0), *_9->getOutput(0), ElementWiseOperation::kSUM);
    auto *_11 = network->addActivation(*_10->getOutput(0), ActivationType::kRELU);

    w         = Weights {DataType::kFLOAT, pWeight, 1024 * 10};
    b         = Weights {DataType::kFLOAT, pWeight, 10};
    auto *_12 = network->addConstant(Dims32 {2, {10, 1024}}, w);
    auto *_13 = network->addMatrixMultiply(*_11->getOutput(0), MatrixOperation::kNONE, *_12->getOutput(0), MatrixOperation::kTRANSPOSE);
    auto *_14 = network->addConstant(Dims32 {2, {1, 10}}, b);
    auto *_15 = network->addElementWise(*_13->getOutput(0), *_14->getOutput(0), ElementWiseOperation::kSUM);

    auto *_16 = network->addSoftMax(*_15->getOutput(0));
    _16->setAxes(1U << 1);

    auto *_17 = network->addTopK(*_16->getOutput(0), TopKOperation::kMAX, 1, 1U << 1);

    network->markOutput(*_17->getOutput(1));

    IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
    if (engineString == nullptr || engineString->size() == 0)
    {
        std::cout << "Failed building serialized engine!" << std::endl;
        return 1;
    }
    std::cout << "Succeeded building serialized engine!" << std::endl;

    delete[] pWeight;
    delete pCalibrator;

    std::cout << "Finish!" << std::endl;
    return 0;
}
