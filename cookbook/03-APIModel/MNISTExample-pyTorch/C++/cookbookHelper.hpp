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
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

// cuda runtime 函数返回值检查
#define ck(call) check(call, __LINE__, __FILE__)

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
            std::cout << std::fixed << std::setprecision(3) << std::setw(6) << pArray[iStart + i] << " ";
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
    double absSum   = double(fabs(pArray[0]));
    double sum2     = double(pArray[0]) * double(pArray[0]);
    double diff     = 0;
    T      maxValue = pArray[0];
    T      minValue = pArray[0];
    for (int i = 1; i < nElement; ++i)
    {
        sum += double(pArray[i]);
        absSum += double(fabs(pArray[i]));
        sum2 += double(pArray[i]) * double(pArray[i]);
        maxValue = pArray[i] > maxValue ? pArray[i] : maxValue;
        minValue = pArray[i] < minValue ? pArray[i] : minValue;
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
        std::cout << std::fixed << std::setprecision(5) << std::setw(8) << pArray[i] << ", ";
    }
    std::cout << std::endl;
    for (int i = nElement - n; i < nElement; ++i)
    {
        std::cout << std::fixed << std::setprecision(5) << std::setw(8) << pArray[i] << ", ";
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

// 计算数据空间的帮助函数
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
// 的打印网络用的帮助函数
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

// 的打印网络用的帮助函数
__inline__ std::string dataTypeToString(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return std::string("FLOAT");
    case DataType::kHALF:
        return std::string("HALF");
    case DataType::kINT8:
        return std::string("INT8");
    case DataType::kINT32:
        return std::string("INT32");
    case DataType::kBOOL:
        return std::string("BOOL");
    default:
        return std::string("Unknown");
    }
}
