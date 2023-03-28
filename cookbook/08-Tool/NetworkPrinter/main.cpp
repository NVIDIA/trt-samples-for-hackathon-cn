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

#include "cookbookHelper.cuh"

using namespace nvinfer1;

const std::string trtFile {"./model.plan"};
static Logger     gLogger(ILogger::Severity::kERROR);

int main()
{
    CHECK(cudaSetDevice(0));
    IBuilder *            builder = createInferBuilder(gLogger);
    INetworkDefinition *  network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    IBuilderConfig *      config  = builder->createBuilderConfig();
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);

    ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims32 {4, {-1, 1, 28, 28}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 1, 28, 28}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {4, 1, 28, 28}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {16, 1, 28, 28}});
    config->addOptimizationProfile(profile);

    float * pW0 = new float[32 * 1 * 5 * 5];
    float * pB0 = new float[32];
    Weights w0 {nvinfer1::DataType::kFLOAT, pW0, 32 * 1 * 5 * 5};
    Weights b0 {nvinfer1::DataType::kFLOAT, pB0, 32};
    auto *  _0 = network->addConvolutionNd(*inputTensor, 32, DimsHW {5, 5}, w0, b0);
    _0->setPaddingNd(Dims32 {2, {2, 2}});

    auto _1 = network->addActivation(*_0->getOutput(0), ActivationType::kRELU);
    auto _2 = network->addPoolingNd(*_1->getOutput(0), PoolingType::kMAX, Dims32 {2, {2, 2}});
    _2->setStrideNd(Dims32 {2, {2, 2}});

    float * pW1 = new float[64 * 32 * 5 * 5];
    float * pB1 = new float[64];
    Weights w1 {nvinfer1::DataType::kFLOAT, pW1, 64 * 32 * 5 * 5};
    Weights b1 {nvinfer1::DataType::kFLOAT, pB1, 64};
    auto *  _3 = network->addConvolutionNd(*_2->getOutput(0), 64, DimsHW {5, 5}, w1, b1);
    _3->setPaddingNd(Dims32 {2, {2, 2}});

    auto _4 = network->addActivation(*_3->getOutput(0), ActivationType::kRELU);
    auto _5 = network->addPoolingNd(*_4->getOutput(0), PoolingType::kMAX, Dims32 {2, {2, 2}});
    _5->setStrideNd(Dims32 {2, {2, 2}});

    auto _6 = network->addShuffle(*_5->getOutput(0));
    _6->setFirstTranspose(Permutation {0, 2, 3, 1});
    _6->setReshapeDimensions(Dims32 {2, {-1, 64 * 7 * 7}});

    float * pW2 = new float[64 * 7 * 7 * 1024];
    float * pB2 = new float[1024];
    Weights w2 {nvinfer1::DataType::kFLOAT, pW2, 64 * 7 * 7 * 1024};
    Weights b2 {nvinfer1::DataType::kFLOAT, pB2, 1024};
    auto    _7  = network->addConstant(Dims32 {2, {64 * 7 * 7, 1024}}, w2);
    auto    _8  = network->addMatrixMultiply(*_6->getOutput(0), MatrixOperation::kNONE, *_7->getOutput(0), MatrixOperation::kNONE);
    auto    _9  = network->addConstant(Dims32 {2, {1, 1024}}, b2);
    auto    _10 = network->addElementWise(*_8->getOutput(0), *_9->getOutput(0), ElementWiseOperation::kSUM);
    auto    _11 = network->addActivation(*_10->getOutput(0), ActivationType::kRELU);

    float * pW3 = new float[1024 * 10];
    float * pB3 = new float[10];
    Weights w3 {nvinfer1::DataType::kFLOAT, pW3, 1024 * 10};
    Weights b3 {nvinfer1::DataType::kFLOAT, pB3, 10};
    auto    _12 = network->addConstant(Dims32 {2, {1024, 10}}, w3);
    auto    _13 = network->addMatrixMultiply(*_11->getOutput(0), MatrixOperation::kNONE, *_12->getOutput(0), MatrixOperation::kNONE);
    auto    _14 = network->addConstant(Dims32 {2, {1, 10}}, b3);
    auto    _15 = network->addElementWise(*_13->getOutput(0), *_14->getOutput(0), ElementWiseOperation::kSUM);

    auto _16 = network->addSoftMax(*_15->getOutput(0));
    _16->setAxes(1 << 1);
    auto _17 = network->addTopK(*_16->getOutput(0), TopKOperation::kMAX, 1, 1 << 1);

    network->markOutput(*_17->getOutput(1));

    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        ILayer *layer = network->getLayer(i);
        std::cout << std::setw(4) << i << std::string("->") << layerTypeToString(layer->getType()) << std::string(",in=") << layer->getNbInputs() << std::string(",out=") << layer->getNbOutputs() << std::string(",") << std::string(layer->getName()) << std::endl;
        for (int j = 0; j < layer->getNbInputs(); ++j)
        {
            ITensor *tensor = layer->getInput(j);
            std::cout << std::string("\tInput  ") << std::setw(2) << j << std::string(":") << shapeToString(tensor->getDimensions()) << std::string(",") << dataTypeToString(tensor->getType()) << std::string(",") << std ::string(tensor->getName()) << std::endl;
        }
        for (int j = 0; j < layer->getNbOutputs(); ++j)
        {
            ITensor *tensor = layer->getOutput(j);
            std::cout << std::string("\tOutput ") << std::setw(2) << j << std::string(":") << shapeToString(tensor->getDimensions()) << std::string(",") << dataTypeToString(tensor->getType()) << std::string(",") << std ::string(tensor->getName()) << std::endl;
        }
    }
    /*
    IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
    if (engineString->size() == 0)
    {
        std::cout << "Failed building serialized engine!" << std::endl;
        delete[] pW0;
        delete[] pB0;
        return 1;
    }
    std::cout << "Succeeded building serialized engine!" << std::endl;
    */
    delete[] pW0;
    delete[] pB0;
    return 0;
}
