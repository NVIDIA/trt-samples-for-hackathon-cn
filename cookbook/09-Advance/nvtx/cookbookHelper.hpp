/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once
#include <NvInfer.h>
#include <cassert>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

// cuda runtime 函数返回值检查
#define CHECK(call) check(call, __LINE__, __FILE__)

inline bool check(cudaError_t e, int iLine, const char *szFile)
{
    if (e != cudaSuccess)
    {
        std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << std::endl;
        return false;
    }
    return true;
}

using namespace nvinfer1;

// Plugin 用到的工具函数
#ifdef DEBUG
    #define WHERE_AM_I()                          \
        do                                        \
        {                                         \
            printf("%14p[%s]\n", this, __func__); \
        } while (0);
#else
    #define WHERE_AM_I()
#endif // ifdef DEBUG

#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))
#define ALIGN_TO(X, Y)    (CEIL_DIVIDE(X, Y) * (Y))

// TensorRT 日志结构体
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
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

// 打印数据用的帮助函数
template<typename T>
void printArrayRecursion(const T *pArray, Dims32 dim, int iDim, int iStart)
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

template<typename T>
void printArrayInfomation(const T *pArray, Dims32 dim, std::string name = std::string(""), bool bPrintArray = false, int n = 10)
{
    // print shape information
    std::cout << std::endl;
    std::cout << name << ": (";
    for (int i = 0; i < dim.nbDims; ++i)
    {
        std::cout << dim.d[i] << ", ";
    }
    std::cout << ")" << std::endl;

    // print statistic information
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

    // print the whole array
    if (bPrintArray)
    {
        printArrayRecursion<T>(pArray, dim, 0, 0);
    }

    return;
}
template void printArrayInfomation(const float *, Dims32, std::string, bool, int);
template void printArrayInfomation(const half *, Dims32, std::string, bool, int);
template void printArrayInfomation(const int *, Dims32, std::string, bool, int);
template void printArrayInfomation(const bool *, Dims32, std::string, bool, int);

// 计算数据尺寸的帮助函数
__inline__ size_t dataTypeToSize(DataType dataType)
{
    switch ((int)dataType)
    {
    case int(DataType::kFLOAT):
        return 4;
    case int(DataType::kHALF):
        return 2;
    case int(DataType::kINT8):
        return 1;
    case int(DataType::kINT32):
        return 4;
    case int(DataType::kBOOL):
        return 1;
    default:
        return 4;
    }
}

// 张量形状转字符串
__inline__ std::string shapeToString(Dims32 dim)
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

// 数据类型转字符串
__inline__ std::string dataTypeToString(DataType dataType)
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
    default:
        return std::string("Unknown");
    }
}

// 张量数据排布转字符串
std::string getFormatString(TensorFormat format)
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
