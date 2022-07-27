/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#define ck(call) check(call, __LINE__, __FILE__)

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

    void log(Severity severity, const char *msg) noexcept override
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
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

static Logger gLogger(ILogger::Severity::kERROR);

__inline__ std::string layerTypeToString(LayerType layerType)
{
    switch (layerType)
    {
    case LayerType::kCONVOLUTION: return std::string("CONVOLUTION");
    case LayerType::kFULLY_CONNECTED: return std::string("FULLY_CONNECTED");
    case LayerType::kACTIVATION: return std::string("ACTIVATION");
    case LayerType::kPOOLING: return std::string("POOLING");
    case LayerType::kLRN: return std::string("LRN");
    case LayerType::kSCALE: return std::string("SCALE");
    case LayerType::kSOFTMAX: return std::string("SOFTMAX");
    case LayerType::kDECONVOLUTION: return std::string("DECONVOLUTION");
    case LayerType::kCONCATENATION: return std::string("CONCATENATION");
    case LayerType::kELEMENTWISE: return std::string("ELEMENTWISE");
    case LayerType::kPLUGIN: return std::string("PLUGIN");
    case LayerType::kUNARY: return std::string("UNARY");
    case LayerType::kPADDING: return std::string("PADDING");
    case LayerType::kSHUFFLE: return std::string("SHUFFLE");
    case LayerType::kREDUCE: return std::string("REDUCE");
    case LayerType::kTOPK: return std::string("TOPK");
    case LayerType::kGATHER: return std::string("GATHER");
    case LayerType::kMATRIX_MULTIPLY: return std::string("MATRIX_MULTIPLY");
    case LayerType::kRAGGED_SOFTMAX: return std::string("RAGGED_SOFTMAX");
    case LayerType::kCONSTANT: return std::string("CONSTANT");
    case LayerType::kRNN_V2: return std::string("RNN_V2");
    case LayerType::kIDENTITY: return std::string("IDENTITY");
    case LayerType::kPLUGIN_V2: return std::string("PLUGIN_V2");
    case LayerType::kSLICE: return std::string("SLICE");
    case LayerType::kSHAPE: return std::string("SHAPE");
    case LayerType::kPARAMETRIC_RELU: return std::string("PARAMETRIC_RELU");
    case LayerType::kRESIZE: return std::string("RESIZE");
    case LayerType::kTRIP_LIMIT: return std::string("TRIP_LIMIT");
    case LayerType::kRECURRENCE: return std::string("RECURRENCE");
    case LayerType::kITERATOR: return std::string("ITERATOR");
    case LayerType::kLOOP_OUTPUT: return std::string("LOOP_OUTPUT");
    case LayerType::kSELECT: return std::string("SELECT");
    case LayerType::kFILL: return std::string("FILL");
    case LayerType::kQUANTIZE: return std::string("QUANTIZE");  // Quantize 层以下为 TensorRT 8 才有的
    case LayerType::kDEQUANTIZE: return std::string("DEQUANTIZE");
    case LayerType::kCONDITION: return std::string("CONDITION");
    case LayerType::kCONDITIONAL_INPUT: return std::string("CONDITIONAL_INPUT");
    case LayerType::kCONDITIONAL_OUTPUT: return std::string("CONDITIONAL_OUTPUT");
    case LayerType::kSCATTER: return std::string("SCATTER");
    case LayerType::kEINSUM: return std::string("EINSUM");
    case LayerType::kASSERTION: return std::string("ASSERTION");
    default: return std::string("Unknown");
    }
}

__inline__ std::string shapeToString(Dims dim)
{
    std::string output("(");
    if (dim.nbDims == 0)
    {
        return output + std::string(")");
    }
    for (int i = 0; i < dim.nbDims - 1; ++i)
    {
        output += std::to_string(dim.d[i]) + std::string(", ");
    }
    output += std::to_string(dim.d[dim.nbDims - 1]) + std::string(")");
    return output;
}

__inline__ std::string dataTypeToString(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT: return std::string("FLOAT");
    case DataType::kHALF: return std::string("HALF");
    case DataType::kINT8: return std::string("INT8");
    case DataType::kINT32: return std::string("INT32");
    case DataType::kBOOL: return std::string("BOOL");
    default: return std::string("Unknown");
    }
}

int main()
{
    ck(cudaSetDevice(0));
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
    auto *  _0 = network->addConvolution(*inputTensor, 32, DimsHW {5, 5}, w0, b0);
    _0->setPaddingNd(Dims32 {2, {2, 2}});

    auto _1 = network->addActivation(*_0->getOutput(0), ActivationType::kRELU);
    auto _2 = network->addPoolingNd(*_1->getOutput(0), PoolingType::kMAX, Dims32 {2, {2, 2}});
    _2->setStrideNd(Dims32 {2, {2, 2}});

    float * pW1 = new float[64 * 32 * 5 * 5];
    float * pB1 = new float[64];
    Weights w1 {nvinfer1::DataType::kFLOAT, pW1, 64 * 32 * 5 * 5};
    Weights b1 {nvinfer1::DataType::kFLOAT, pB1, 64};
    auto *  _3 = network->addConvolution(*_2->getOutput(0), 64, DimsHW {5, 5}, w1, b1);
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
        auto layer = network->getLayer(i);
        std::cout << std::setw(4) << i << std::string("->") << layerTypeToString(layer->getType()) << std::string(",in=") << layer->getNbInputs() << std::string(",out=") << layer->getNbOutputs() << std::string(",") << std::string(layer->getName()) << std::endl;
        for (int j = 0; j < layer->getNbInputs(); ++j)
        {
            auto tensor = layer->getInput(j);
            std::cout << std::string("\tInput  ") << std::setw(2) << j << std::string(":") << shapeToString(tensor->getDimensions()) << std::string(",") << dataTypeToString(tensor->getType()) << std::string(",") << std ::string(tensor->getName()) << std::endl;
        }
        for (int j = 0; j < layer->getNbOutputs(); ++j)
        {
            auto tensor = layer->getOutput(j);
            std::cout << std::string("\tOutput ") << std::setw(2) << j << std::string(":") << shapeToString(tensor->getDimensions()) << std::string(",") << dataTypeToString(tensor->getType()) << std::string(",") << std ::string(tensor->getName()) << std::endl;
        }
    }

    IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
    if (engineString->size() == 0)
    {
        std::cout << "Failed building serialized engine!" << std::endl;
        return 1;
    }
    std::cout << "Succeeded building serialized engine!" << std::endl;

    delete[] pW0;
    delete[] pB0;

    return 0;
}
