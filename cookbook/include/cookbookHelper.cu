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

#include "cookbookHelper.cuh"

void loadPluginFile(const std::string &path)
{
#ifdef _MSC_VER
    void *handle = LoadLibrary(path.c_str());
#else
    int32_t flags {RTLD_LAZY};
    void   *handle = dlopen(path.c_str(), flags);
#endif
    if (handle == nullptr)
    {
#ifdef _MSC_VER
        std::cout << "Could not load plugin library: " << path << std::endl;
#else
        std::cout << "Could not load plugin library: " << path << ", due to: " << dlerror() << std::endl;
#endif
    }
}

template<typename T>
__global__ void printGPUKernel(T const *const in, int const n)
{
    printf("\n");
    for (int i = 0; i < n; ++i)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            printf("%4d:%.3f,", i, in[i]);
        }
        else if constexpr (std::is_same_v<T, half>)
        {
            printf("%4d:%.3f,", i, float(in[i]));
        }
        else if constexpr (std::is_same_v<T, int>)
        {
            printf("%d:%d,", i, in[i]);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
            printf("%d:%d,", i, int(in[i]));
        }
        else
        {
            printf("[printGPUKernel]Data type error, might be void?");
        }
    }
    printf("\n");
    return;
}

template<typename T>
void printGPU(T const *const in, int const n, cudaStream_t stream)
{
    cudaDeviceSynchronize();
    printf("[printGPU]in=%p, n=%d, stream=%d\n", in, n, stream);
    if (!in)
    {
        printGPUKernel<<<1, 1, 0, stream>>>(in, n);
    }
    cudaDeviceSynchronize();
}

// Print the shape of a TensorRT tensor
void printShape(Dims64 &dim)
{
    std::cout << "[";
    for (int i = 0; i < dim.nbDims; ++i)
    {
        std::cout << dim.d[i] << ", ";
    }
    std::cout << "]" << std::endl;
    return;
}

// Print data in the array
template<typename T>
void printArrayRecursion(const T *pArray, Dims64 dim, int iDim, int iStart)
{
    if (iDim == dim.nbDims - 1)
    {
        for (int i = 0; i < dim.d[iDim]; ++i)
        {
            std::cout << std::fixed << std::setprecision(3) << std::setw(6) << double(pArray[iStart + i]) << " ";
        }
    }
    else
    {
        int nElement = 1;
        for (int i = iDim + 1; i < dim.nbDims; ++i)
        {
            nElement *= dim.d[i];
        }
        for (int i = 0; i < dim.d[iDim]; ++i)
        {
            printArrayRecursion<T>(pArray, dim, iDim + 1, iStart + i * nElement);
        }
    }
    std::cout << std::endl;
    return;
}

// Get the size in byte of a TensorRT data type
size_t dataTypeToSize(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT8:
        return 1;
    case DataType::kINT32:
        return 4;
    case DataType::kBOOL:
        return 1;
    case DataType::kUINT8:
        return 1;
    case DataType::kFP8:
        return 1;
    case DataType::kINT64:
        return 8;
    default:
        return 4;
    }
}

// Get the string of a TensorRT shape
std::string shapeToString(Dims64 dim)
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

// Get the string of a TensorRT data type
std::string dataTypeToString(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return std::string("FP32 ");
    case DataType::kHALF:
        return std::string("FP16 ");
    case DataType::kINT8:
        return std::string("INT8 ");
    case DataType::kINT32:
        return std::string("INT32");
    case DataType::kBOOL:
        return std::string("BOOL ");
    case DataType::kUINT8:
        return std::string("UINT8");
    case DataType::kFP8:
        return std::string("FP8  ");
    case DataType::kINT64:
        return std::string("INT64");
    default:
        return std::string("Unknown");
    }
}

// Get the string of a TensorRT data format
std::string formatToString(TensorFormat format)
{
    switch (format)
    {
    case TensorFormat::kLINEAR:
        return std::string("LINE ");
    case TensorFormat::kCHW2:
        return std::string("CHW2 ");
    case TensorFormat::kHWC8:
        return std::string("HWC8 ");
    case TensorFormat::kCHW4:
        return std::string("CHW4 ");
    case TensorFormat::kCHW16:
        return std::string("CHW16");
    case TensorFormat::kCHW32:
        return std::string("CHW32");
    case TensorFormat::kHWC:
        return std::string("HWC  ");
    case TensorFormat::kDLA_LINEAR:
        return std::string("DLINE");
    case TensorFormat::kDLA_HWC4:
        return std::string("DHWC4");
    case TensorFormat::kHWC16:
        return std::string("HWC16");
    default: return std::string("None ");
    }
}

// Get the string of a TensorRT layer kind
std::string layerTypeToString(LayerType layerType)
{
    switch (layerType)
    {
    case LayerType::kCONVOLUTION: return std::string("CONVOLUTION");
    case LayerType::kCAST: return std::string("CAST");
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
    case LayerType::kQUANTIZE: return std::string("QUANTIZE");
    case LayerType::kDEQUANTIZE: return std::string("DEQUANTIZE");
    case LayerType::kCONDITION: return std::string("CONDITION");
    case LayerType::kCONDITIONAL_INPUT: return std::string("CONDITIONAL_INPUT");
    case LayerType::kCONDITIONAL_OUTPUT: return std::string("CONDITIONAL_OUTPUT");
    case LayerType::kSCATTER: return std::string("SCATTER");
    case LayerType::kEINSUM: return std::string("EINSUM");
    case LayerType::kASSERTION: return std::string("ASSERTION");
    case LayerType::kONE_HOT: return std::string("ONE_HOT");
    case LayerType::kNON_ZERO: return std::string("NON_ZERO");
    case LayerType::kGRID_SAMPLE: return std::string("GRID_SAMPLE");
    case LayerType::kNMS: return std::string("NMS");
    case LayerType::kREVERSE_SEQUENCE: return std::string("REVERSE_SEQUENCE");
    case LayerType::kNORMALIZATION: return std::string("NORMALIZATION");
    case LayerType::kPLUGIN_V3: return std::string("PLUGIN_V3");
    default: return std::string("Unknown");
    }
}

// Get the string of a TensorRT tensor location
std::string locationToString(TensorLocation location)
{
    switch (location)
    {
    case TensorLocation::kHOST:
        return std::string("HOST");
    case TensorLocation::kDEVICE:
        return std::string("DEVICE");
    default: return std::string("None");
    }
}

template<typename T>
void printArrayInformation(
    T const *const     pArray,
    std::string const &name,
    Dims64 const      &dim,
    bool const         bPrintInformation,
    bool const         bPrintArray,
    int const          n)
{
    // Print shape information
    //int nElement = std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<>());
    std::cout << std::endl;
    std::cout << name << ": " << typeid(T).name() << ", " << shapeToString(dim) << std::endl;

    // Print statistic information of the array
    if (bPrintInformation)
    {
        int nElement = 1; // number of elements with batch dimension
        for (int i = 0; i < dim.nbDims; ++i)
        {
            nElement *= dim.d[i];
        }

        double sum      = double(pArray[0]);
        double absSum   = double(fabs(double(pArray[0])));
        double sum2     = double(pArray[0]) * double(pArray[0]);
        double diff     = 0.0;
        double maxValue = double(pArray[0]);
        double minValue = double(pArray[0]);
        for (int i = 1; i < nElement; ++i)
        {
            sum += double(pArray[i]);
            absSum += double(fabs(double(pArray[i])));
            sum2 += double(pArray[i]) * double(pArray[i]);
            maxValue = double(pArray[i]) > maxValue ? double(pArray[i]) : maxValue;
            minValue = double(pArray[i]) < minValue ? double(pArray[i]) : minValue;
            diff += abs(double(pArray[i]) - double(pArray[i - 1]));
        }
        double mean = sum / nElement;
        double var  = sum2 / nElement - mean * mean;

        std::cout << "absSum=" << std::fixed << std::setprecision(4) << std::setw(7) << absSum << ",";
        std::cout << "mean=" << std::fixed << std::setprecision(4) << std::setw(7) << mean << ",";
        std::cout << "var=" << std::fixed << std::setprecision(4) << std::setw(7) << var << ",";
        std::cout << "max=" << std::fixed << std::setprecision(4) << std::setw(7) << maxValue << ",";
        std::cout << "min=" << std::fixed << std::setprecision(4) << std::setw(7) << minValue << ",";
        std::cout << "diff=" << std::fixed << std::setprecision(4) << std::setw(7) << diff << ",";
        std::cout << std::endl;

        // print first n element and last n element
        for (int i = 0; i < n; ++i)
        {
            std::cout << std::fixed << std::setprecision(5) << std::setw(8) << double(pArray[i]) << ", ";
        }
        std::cout << std::endl;
        for (int i = nElement - n; i < nElement; ++i)
        {
            std::cout << std::fixed << std::setprecision(5) << std::setw(8) << double(pArray[i]) << ", ";
        }
        std::cout << std::endl;
    }

    // print the data of the array
    if (bPrintArray)
    {
        printArrayRecursion<T>(pArray, dim, 0, 0);
    }

    return;
}
template void printArrayInformation(float const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);
template void printArrayInformation(half const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);
template void printArrayInformation(char const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);
template void printArrayInformation(int const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);
template void printArrayInformation(bool const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);
template void printArrayInformation(int8_t const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);
template void printArrayInformation(int64_t const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);

void printNetwork(INetworkDefinition *network)
{
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        ILayer           *layer = network->getLayer(i);
        std::stringstream ss;
        ss << std::setw(4) << i << std::setw(0) << "->[" << layerTypeToString(layer->getType()) << "]->" << layer->getName();
        std::cout << ss.str() << std::endl;
        ss.str("");
        for (int j = 0; j < layer->getNbInputs(); ++j)
        {
            ITensor *tensor = layer->getInput(j);
            ss << "    In " << std::setw(2) << j << std::setw(0) << ":";
            if (tensor == nullptr)
            {
                ss << "None";
            }
            else
            {
                ss << shapeToString(tensor->getDimensions()) << "," << dataTypeToString(tensor->getType()) << "," << locationToString(tensor->getLocation()) << "," << tensor->getName();
                if (tensor->isNetworkInput())
                {
                    ss << " <-(NETWORK_INPUT)";
                }
            }
            std::cout << ss.str() << std::endl;
            ss.str("");
        }
        for (int j = 0; j < layer->getNbOutputs(); ++j)
        {
            ITensor *tensor = layer->getOutput(j);
            ss << "    Out" << std::setw(2) << j << std::setw(0) << ":";
            if (tensor == nullptr)
            {
                ss << "None";
            }
            else
            {
                ss << shapeToString(tensor->getDimensions()) << "," << dataTypeToString(tensor->getType()) << "," << locationToString(tensor->getLocation()) << "," << tensor->getName();
                if (tensor->isNetworkOutput())
                {
                    ss << " <-(NETWORK_OUTPUT)";
                }
            }
            std::cout << ss.str() << std::endl;
            ss.str("");
        }
    }
}

std::vector<ITensor *> buildMnistNetwork(IBuilderConfig *config, INetworkDefinition *network, IOptimizationProfile *profile)
{
    ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims64 {4, {-1, 1, 28, 28}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims64 {4, {1, 1, 28, 28}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims64 {4, {2, 1, 28, 28}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims64 {4, {4, 1, 28, 28}});
    config->addOptimizationProfile(profile);

    float *pWorkspace = new float[64 * 7 * 7 * 1024]; // a pointer for dummy weights

    Weights w {nvinfer1::DataType::kFLOAT, pWorkspace, 32 * 1 * 5 * 5};
    Weights b {nvinfer1::DataType::kFLOAT, pWorkspace, 32};
    auto   *_0 = network->addConvolutionNd(*inputTensor, 32, DimsHW {5, 5}, w, b);
    _0->setName("Convolution1");
    _0->setPaddingNd(Dims64 {2, {2, 2}});

    auto _1 = network->addActivation(*_0->getOutput(0), ActivationType::kRELU);
    _1->setName("Activation1");
    auto _2 = network->addPoolingNd(*_1->getOutput(0), PoolingType::kMAX, Dims64 {2, {2, 2}});
    _2->setName("Pooling1");
    _2->setStrideNd(Dims64 {2, {2, 2}});

    w        = Weights {nvinfer1::DataType::kFLOAT, pWorkspace, 64 * 32 * 5 * 5};
    b        = Weights {nvinfer1::DataType::kFLOAT, pWorkspace, 64};
    auto *_3 = network->addConvolutionNd(*_2->getOutput(0), 64, DimsHW {5, 5}, w, b);
    _3->setName("Convolution2");
    _3->setPaddingNd(Dims64 {2, {2, 2}});

    auto _4 = network->addActivation(*_3->getOutput(0), ActivationType::kRELU);
    _4->setName("Activation2");
    auto _5 = network->addPoolingNd(*_4->getOutput(0), PoolingType::kMAX, Dims64 {2, {2, 2}});
    _5->setName("Pooling2");
    _5->setStrideNd(Dims64 {2, {2, 2}});

    auto _6 = network->addShuffle(*_5->getOutput(0));
    _6->setName("Shuffle");
    _6->setReshapeDimensions(Dims64 {2, {-1, 64 * 7 * 7}});

    w       = Weights {nvinfer1::DataType::kFLOAT, pWorkspace, 64 * 7 * 7 * 1024};
    b       = Weights {nvinfer1::DataType::kFLOAT, pWorkspace, 1024};
    auto _7 = network->addConstant(Dims64 {2, {64 * 7 * 7, 1024}}, w);
    _7->setName("MatrixMultiplication1Weight");
    auto _8 = network->addMatrixMultiply(*_6->getOutput(0), MatrixOperation::kNONE, *_7->getOutput(0), MatrixOperation::kNONE);
    _8->setName("MatrixMultiplication1");
    auto _9 = network->addConstant(Dims64 {2, {1, 1024}}, b);
    _9->setName("ConstantBias1");
    auto _10 = network->addElementWise(*_8->getOutput(0), *_9->getOutput(0), ElementWiseOperation::kSUM);
    _10->setName("AddBias1");
    auto _11 = network->addActivation(*_10->getOutput(0), ActivationType::kRELU);
    _11->setName("Activation3");

    w        = Weights {nvinfer1::DataType::kFLOAT, pWorkspace, 1024 * 10};
    b        = Weights {nvinfer1::DataType::kFLOAT, pWorkspace, 10};
    auto _12 = network->addConstant(Dims64 {2, {1024, 10}}, w);
    _12->setName("MatrixMultiplication2Weight");
    auto _13 = network->addMatrixMultiply(*_11->getOutput(0), MatrixOperation::kNONE, *_12->getOutput(0), MatrixOperation::kNONE);
    _13->setName("MatrixMultiplication2");
    auto _14 = network->addConstant(Dims64 {2, {1, 10}}, b);
    _14->setName("ConstantBias2");
    auto _15 = network->addElementWise(*_13->getOutput(0), *_14->getOutput(0), ElementWiseOperation::kSUM);
    _15->setName("AddBias2");
    auto _16 = network->addSoftMax(*_15->getOutput(0));
    _16->setName("SoftMax");
    _16->setAxes(1 << 1);
    auto _17 = network->addTopK(*_16->getOutput(0), TopKOperation::kMAX, 1, 1 << 1);
    _17->setName("TopK");

    return std::vector<ITensor *> {_17->getOutput(1)}; // ITensor is not copyable
}
