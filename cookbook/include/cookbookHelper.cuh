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

#ifndef COOKBOOKHELPER_CUH
#define COOKBOOKHELPER_CUH

#include "cnpy.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h> // load ".so" (plugin)
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

// result value check of cuda runtime
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

#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))
#define ALIGN_TO(X, Y)    (CEIL_DIVIDE(X, Y) * (Y))

void loadPluginFile(const std::string &path);

template<typename T>
__global__ static void printGPUKernel(T const *const in, int const n);

// Do not enable this function here, it leads to many errors about cub
template<typename T>
void printGPU(T const *const in, int const n = 10, cudaStream_t stream = 0);

// TensorRT journal
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

// Simplified FileStreamReader for loading engine from file
class FileStreamReader final : public nvinfer1::IStreamReader
{
public:
    FileStreamReader(std::string file):
        mFile {std::ifstream(file, std::ios::binary)} {}

    ~FileStreamReader() final
    {
        mFile.close();
    }

    int64_t read(void *dest, int64_t bytes) final // necessary API
    {
        if (!mFile.good())
        {
            return -1;
        }
        mFile.read(static_cast<char *>(dest), bytes);
        return mFile.gcount();
    }

private:
    std::ifstream mFile;
};

// Print the shape of a TensorRT tensor
void printShape(Dims64 &dim);

// Print data in the array
template<typename T>
void printArrayRecursion(const T *pArray, Dims64 dim, int iDim, int iStart);

// Get the size in byte of a TensorRT data type
size_t dataTypeToSize(DataType dataType);

// Get the string of a TensorRT shape
std::string shapeToString(Dims64 dim);

// Get the string of a TensorRT data type
std::string dataTypeToString(DataType dataType);

// Get the string of a TensorRT data format
std::string formatToString(TensorFormat format);

// Get the string of a TensorRT layer kind
std::string layerTypeToString(LayerType layerType);

// Get the string of a TensorRT tensor location
std::string locationToString(TensorLocation location);

template<typename T>
void printArrayInformation(
    T const *const     pArray,
    std::string const &name,
    Dims64 const      &dim,
    bool const         bPrintInformation = false,
    bool const         bPrintArray       = false,
    int const          n                 = 10);

void printNetwork(INetworkDefinition *network);

std::vector<ITensor *> buildMnistNetwork(IBuilderConfig *config, INetworkDefinition *network, IOptimizationProfile *profile);

// plugin debug function
#ifdef DEBUG
    #define WHERE_AM_I() printf("%14p[%s]\n", this, __func__);
    #define PRINT_FORMAT_COMBINATION()                                    \
        do                                                                \
        {                                                                 \
            std::cout << "    pos=" << pos << ":[";                       \
            for (int i = 0; i < nbInputs + nbOutputs; ++i)                \
            {                                                             \
                std::cout << dataTypeToString(inOut[i].desc.type) << ","; \
            }                                                             \
            std::cout << "],[";                                           \
            for (int i = 0; i < nbInputs + nbOutputs; ++i)                \
            {                                                             \
                std::cout << formatToString(inOut[i].desc.format) << ","; \
            }                                                             \
            std::cout << "]->";                                           \
            std::cout << "res=" << res << std::endl;                      \
        } while (0);

#else
    #define WHERE_AM_I()
    #define PRINT_FORMAT_COMBINATION()
#endif // ifdef DEBUG

#endif // COOKBOOKHELPER_CUH
