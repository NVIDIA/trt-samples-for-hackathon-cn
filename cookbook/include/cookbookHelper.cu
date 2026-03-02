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
    case DataType::kINT4:
        return 1; // 0.5
    case DataType::kFP4:
        return 1; // 0.5
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
    auto _17 = network->addTopK(*_16->getOutput(0), TopKOperation::kMAX, 1, 1 << 1, nvinfer1::DataType::kINT32);
    _17->setName("TopK");

    return std::vector<ITensor *> {_17->getOutput(1)}; // ITensor is not copyable
}

/*
// Move from TensorRT-LLM
// TODO: enable them later

template <typename T>
void printArrayInfo(T const* ptr, uint64_t nElement = 1, std::string name = "", bool const bPrintElement = false)
{
    if (ptr == nullptr)
    {
        TLLM_LOG_WARNING("%s is an nullptr, skip!", name.c_str());
        return;
    }
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    bool const isDevicePtr = (getPtrCudaMemoryType(ptr) == cudaMemoryTypeDevice);
    size_t sizeInByte = sizeof(T) * nElement;
    TLLM_LOG_TRACE("addr=%p, location=%s, sizeof(T)=%lu, nElement=%d, sizeInByte=%lu\n", ptr,
        (isDevicePtr ? "Device" : "Host"), sizeof(T), nElement, sizeInByte);
    T* tmp = const_cast<T*>(ptr);
    std::vector<T> tmpVec; // For device pointer
    if (isDevicePtr)
    {
        tmpVec.resize(nElement);
        tmp = tmpVec.data(); // Note `data()` is not supported for vector<bool>
        check_cuda_error(cudaMemcpy(tmp, ptr, sizeInByte, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }

    size_t nInf = 0;
    size_t nNaN = 0;
    size_t nZero = 0;
    double sum = 0.0;
    double sqrSum = 0.0;
    double absSum = 0.0;
    float allMax = -1.0e6f;
    float allMin = 1.0e6f;
    float allSad = 0.0f; // Sum Abs of Difference, to distinguish A and its transpose
    float old = 0.0f;
    for (uint64_t i = 0; i < nElement; i++)
    {
        float val = (float) tmp[i];

        if (std::isinf(val))
        {
            nInf++;
            continue;
        }
        if (std::isnan(val))
        {
            nNaN++;
            continue;
        }
        nZero += (val == 0.0f);
        sum += val;
        sqrSum += val * val;
        absSum += expf(val);
        allMax = std::max(allMax, val);
        allMin = std::min(allMin, val);
        allSad += abs(val - old);
        old = val;
    }
    float avg = sum / nElement;
    float std = sqrtf(sqrSum / nElement - avg * avg);

    TLLM_LOG_INFO("%s", name.c_str());
    TLLM_LOG_INFO("size=%u, nInf=%zu, nNaN=%zu, nZero=%zu", nElement, nInf, nNaN, nZero);
    TLLM_LOG_INFO("avg=%f, absSum: %f, std=%f, max=%f, min=%f, sad=%f", avg, absSum, std, allMax, allMin, allSad);

    if (bPrintElement)
    {
        uint64_t constexpr nHead = 5;
        std::stringstream ss;
        ss << std::setw(10) << std::fixed << std::setprecision(3);
        for (uint64_t i = 0; i < std::min(nElement, nHead); ++i)
        {
            ss << (float) tmp[i] << ", ";
        }
        if (nElement > nHead)
        {
            ss << " ... ";
            for (uint64_t i = nElement - nHead; i < nElement; ++i)
            {
                ss << (float) tmp[i] << ", ";
            }
        }
        TLLM_LOG_INFO("%s", ss.str().c_str());
    }
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
}

template void printArrayInfo(float const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
template void printArrayInfo(half const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
#ifdef ENABLE_BF16
template void printArrayInfo(__nv_bfloat16 const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
#endif
#ifdef ENABLE_FP8
template void printArrayInfo(__nv_fp8_e4m3 const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
#endif
#ifdef ENABLE_FP4
template void printArrayInfo(__nv_fp4_e2m1 const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
#endif
template void printArrayInfo(uint32_t const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
template void printArrayInfo(uint64_t const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
template void printArrayInfo(int const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
template void printArrayInfo(uint8_t const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);

template <typename T>
void printToStream(T const* ptr, int const nElement, FILE* strm)
{
    bool const split_rows = (strm == stdout);
    if (ptr == nullptr)
    {
        TLLM_LOG_WARNING("Nullptr, skip!\n");
        return;
    }
    std::vector<T> tmp(nElement, 0);
    check_cuda_error(cudaMemcpy(tmp.data(), ptr, sizeof(T) * nElement, cudaMemcpyDeviceToHost));
    for (int i = 0; i < nElement; ++i)
    {
        fprintf(strm, "%f, ", static_cast<float>(tmp[i]));
        if (split_rows && ((i + 1) % 10) == 0)
            fprintf(strm, "\n");
    }
    if (!split_rows || (nElement % 10) != 0)
    {
        fprintf(strm, "\n");
    }
}

template <typename T>
void printToScreen(T const* ptr, int const nElement)
{
    printToStream(ptr, nElement, stdout);
}

template <typename T>
void print2dToStream(T const* ptr, int const nRow, int const nCol, int const nStride, FILE* strm)
{
    if (ptr == nullptr)
    {
        TLLM_LOG_WARNING("Nullptr, skip!\n");
        return;
    }
    for (int ri = 0; ri < nRow; ++ri)
    {
        T const* tmp = ptr + ri * nStride;
        printToStream(tmp, nCol, strm);
    }
    fprintf(strm, "\n");
}

template <typename T>
void print2dToScreen(T const* ptr, int const nRow, int const nCol, int const nStride)
{
    print2dToStream(ptr, nRow, nCol, nStride, stdout);
}

template <typename T>
void print2dToFile(std::string fname, T const* ptr, int const nRow, int const nCol, int const nStride)
{
    FILE* fp = fopen(fname.c_str(), "wt");
    if (fp != nullptr)
    {
        print2dToStream(ptr, nRow, nCol, nStride, fp);
        fclose(fp);
    }
}

__host__ __device__ inline void print_float_(float x)
{
    printf("%7.3f ", x);
}

__host__ __device__ inline void print_element_(float x)
{
    print_float_(x);
}

__host__ __device__ inline void print_element_(half x)
{
    print_float_((float) x);
}

#ifdef ENABLE_BF16
__host__ __device__ inline void print_element_(__nv_bfloat16 x)
{
    print_float_((float) x);
}
#endif

#ifdef ENABLE_FP8
__host__ __device__ inline void print_element_(__nv_fp8_e4m3 x)
{
    print_float_((float) x);
}
#endif

__host__ __device__ inline void print_element_(bool ui)
{
    printf("%7" PRIu32 " ", (unsigned int) ui);
}

__host__ __device__ inline void print_element_(uint8_t ui)
{
    printf("%7" PRIu32 " ", (unsigned int) ui);
}

__host__ __device__ inline void print_element_(uint32_t ul)
{
    printf("%7" PRIu32 " ", ul);
}

__host__ __device__ inline void print_element_(uint64_t ull)
{
    printf("%7" PRIu64 " ", ull);
}

__host__ __device__ inline void print_element_(int32_t il)
{
    printf("%7" PRId32 " ", il);
}

__host__ __device__ inline void print_element_(int64_t ill)
{
    printf("%7" PRId64 " ", ill);
}

template <typename T>
__host__ __device__ inline void print_elements(T const* ptr, int nRow, int nCol, int nStride)
{
    for (int iRow = -1; iRow < nRow; ++iRow)
    {
        if (iRow >= 0)
        {
            printf("%07d|", iRow);
        }
        else
        {
            printf("       |"); // heading row
        }
        for (int iCol = 0; iCol < nCol; iCol += 1)
        {
            if (iRow >= 0)
            {
                print_element_(ptr[iRow * nStride + iCol]);
            }
            else
            {
                printf("%7d|", iCol); // heading colume
            }
        }
        printf("\n");
    }
    printf("\n");
}

template <typename T>
inline void printMatrix(T const* ptr, int nRow, int nCol, int nStride)
{
    // `nRow` is length of row dimension
    // `nStride` is length of column dimension
    // `nCol` (<= nStride) is length for print per row
    if (ptr == nullptr)
    {
        TLLM_LOG_WARNING("Nullptr, skip!\n");
        return;
    }
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    bool const isDevicePtr = (getPtrCudaMemoryType(ptr) == cudaMemoryTypeDevice);
    size_t sizeInByte = sizeof(T) * nRow * nStride;
    TLLM_LOG_TRACE("addr=%p, location=%s, sizeof(T)=%lu, nRow=%d, nStride=%d, sizeInByte=%lu\n", ptr,
        (isDevicePtr ? "Device" : "Host"), sizeof(T), nRow, nStride, sizeInByte);
    if (isDevicePtr)
    {
        std::vector<T> tmpVec;
        tmpVec.resize(nRow * nStride);
        T* tmp = tmpVec.data(); // Note `data()` is not supported for vector<bool>
        check_cuda_error(cudaMemcpy(tmp, ptr, sizeInByte, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
        print_elements(tmp, nRow, nCol, nStride);
    }
    else
    {
        print_elements(ptr, nRow, nCol, nStride);
    }
}

template void printMatrix(float const* ptr, int nRow, int nCol, int nStride);
template void printMatrix(half const* ptr, int nRow, int nCol, int nStride);
#ifdef ENABLE_BF16
template void printMatrix(__nv_bfloat16 const* ptr, int nRow, int nCol, int nStride);
#endif
#ifdef ENABLE_FP8
template void printMatrix(__nv_fp8_e4m3 const* ptr, int nRow, int nCol, int nStride);
#endif
template void printMatrix(uint32_t const* ptr, int nRow, int nCol, int nStride);
template void printMatrix(uint64_t const* ptr, int nRow, int nCol, int nStride);
template void printMatrix(int const* ptr, int nRow, int nCol, int nStride);
template void printMatrix(uint8_t const* ptr, int nRow, int nCol, int nStride);

template <typename T>
__device__ inline void printMatrixDevice(T const* ptr, int nRow, int nCol, int nStride)
{
    // `nRow` is length of row dimension
    // `nStride` is length of column dimension
    // `nCol` (<= nStride) is length for print per row
    // Can be called inside kernels by one single thread
    if (ptr == nullptr)
    {
        printf("Nullptr, skip!\n");
        return;
    }
    size_t sizeInByte = sizeof(T) * nRow * nStride;
    printf("addr=%p, sizeof(T)=%lu, nRow=%d, nStride=%d, sizeInByte=%lu\n", ptr, sizeof(T), nRow, nStride, sizeInByte);
    print_elements(ptr, nRow, nCol, nStride);
}

template __device__ void printMatrixDevice(float const* ptr, int nRow, int nCol, int nStride);
template __device__ void printMatrixDevice(half const* ptr, int nRow, int nCol, int nStride);
#ifdef ENABLE_BF16
template __device__ void printMatrixDevice(__nv_bfloat16 const* ptr, int nRow, int nCol, int nStride);
#endif
#ifdef ENABLE_FP8
template __device__ void printMatrixDevice(__nv_fp8_e4m3 const* ptr, int nRow, int nCol, int nStride);
#endif
template __device__ void printMatrixDevice(uint32_t const* ptr, int nRow, int nCol, int nStride);
template __device__ void printMatrixDevice(uint64_t const* ptr, int nRow, int nCol, int nStride);
template __device__ void printMatrixDevice(int const* ptr, int nRow, int nCol, int nStride);
template __device__ void printMatrixDevice(uint8_t const* ptr, int nRow, int nCol, int nStride);

*/
