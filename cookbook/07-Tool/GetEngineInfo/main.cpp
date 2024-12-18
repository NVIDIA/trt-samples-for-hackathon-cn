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

using namespace nvinfer1;

int constexpr iDevice {0};
int64_t constexpr headerSize {32};

#define PRINT(name)                                                                    \
    do                                                                                 \
    {                                                                                  \
        printf("%-28s:%16ld <->%16ld\n", #name, int64_t(eci.name), int64_t(rci.name)); \
    } while (0);

struct Plan
{
    struct Section
    {
        uint32_t    type;
        void const *start;
        int64_t     size;
    };
    std::vector<Section> sections;
};

struct PlanEntry
{
    uint32_t type {0xdebac1e};
    uint32_t padEntry {};
    int64_t  offset {};
    int64_t  size {};
};

struct CudaInfo // Use name from TRT, different from CUDA prop
{
    int32_t major {0};
    int32_t minor {0};
    int32_t maxCoreClockRate {0};
    int32_t maxMemoryClockRate {0};
    int32_t memoryBusWidth {0};
    size_t  l2CacheSize {0};
    size_t  maxPersistentL2CacheSize {0};
    int32_t sharedMemPerBlock {0};
    int32_t sharedMemPerMultiprocessor {0};
    int32_t textureAlignment {0};
    int32_t multiProcessorCount {0};
    bool    integrated {false};
    int32_t maxThreadsPerBlock {0};
    int32_t maxGridDimX {0};
    int32_t maxGridDimY {0};
    int32_t maxGridDimZ {0};
    int64_t totalGlobalMem {0};
    int32_t maxTexture1DLinear {0};
};

template<typename T>
T readAndMove(uint8_t const **p, const char *name = nullptr)
{
    T const value {*reinterpret_cast<T const *>(*p)};
    if (name != nullptr)
    {
        printf("%-28s:%16ld\n", name, int64_t(value));
    }
    *p += sizeof(T);
    return value;
}

int main(int argv, char **argc)
{
    cudaSetDevice(0);
    if (argv != 2 || argc[1] == nullptr)
    {
        printf("Need a TensorRT engine file name.\n");
        return 1;
    }

    std::string trtFile = std::string(argc[1]);
    if (access(trtFile.c_str(), F_OK) != 0)
    {
        printf("Invalid TensorRT engine file name.\n");
        return 1;
    }

    FileStreamReader filestream(trtFile);

    std::vector<uint8_t> planMemory(headerSize, 0);
    filestream.read(planMemory.data(), headerSize);
    uint8_t const  *p_data = reinterpret_cast<uint8_t const *>(planMemory.data());
    uint8_t const **p      = &p_data; // to move pointer during reading, we need **

    printf("================================================================ Current TensorRT\n");
    printf("Major                       :%16d\n", NV_TENSORRT_MAJOR);
    printf("Minor                       :%16d\n", NV_TENSORRT_MINOR);
    printf("Patch                       :%16d\n", NV_TENSORRT_PATCH);
    printf("Build                       :%16d\n", NV_TENSORRT_BUILD);
    printf("TRT-Python                  :%16s\n", "                ");                          // Just keep alignment with Python version
    printf("================================================================ Engine header\n"); // might change in the future
    printf("HeaderSize                  :%16ld\n", headerSize);
    readAndMove<uint32_t>(p, "MagicTag");
    readAndMove<uint32_t>(p, "SerializationVersion");
    int64_t nEntry   = readAndMove<uint64_t>(p, "nEntry");
    int64_t planSize = readAndMove<uint64_t>(p, "PlanTotalSize");
    int8_t  trtMajor = readAndMove<uint8_t>(p, "TRT.Major");
    int8_t  trtMinor = readAndMove<uint8_t>(p, "TRT.Minor");
    readAndMove<uint8_t>(p, "TRT.Patch");
    readAndMove<uint8_t>(p, "TRT.Build");
    readAndMove<uint32_t>(p, "Pad");

    planMemory.resize(planSize);
    filestream.read(planMemory.data() + headerSize, planSize - headerSize);
    p_data = static_cast<uint8_t const *>(planMemory.data());

    // Keep this part since we may want to deserialize other sections like plugins or weights
    auto plan = std::make_unique<Plan>();

    auto   *entries = reinterpret_cast<PlanEntry const *>(p_data + headerSize);
    int64_t cursor  = headerSize + nEntry * sizeof(PlanEntry);

    std::transform(entries, entries + nEntry, std::back_inserter(plan->sections), [&](PlanEntry const &e)
                   {
        cursor = e.offset + e.size;
        return Plan::Section{e.type, p_data + e.offset, e.size}; });

    cursor = (cursor + 7) / 8 * 8 + sizeof(uint32_t); // add tail padding, now `cursor` should equals to planSize

    std::vector<Plan::Section> engineSections;
    std::copy_if(plan->sections.begin(), plan->sections.end(), std::back_inserter(engineSections), [](Plan::Section const &bs)
                 { return bs.type == 0x454E474E; }); // we only want Engine section

    p_data = static_cast<uint8_t const *>(engineSections[0].start);
    p      = &p_data;

    printf("================================================================ Engine data\n");
    readAndMove<uint32_t>(p, "MagicTag");
    if (!(trtMajor >= 10 && trtMinor >= 6))
    {
        readAndMove<uint32_t>(p, "SafeVersion");
    }
    readAndMove<uint32_t>(p, "StdVersion");
    readAndMove<uint32_t>(p, "HashRead");
    readAndMove<uint64_t>(p, "SizeRead");
    readAndMove<uint32_t>(p);
    readAndMove<uint8_t>(p, "TRT.Major");
    readAndMove<uint32_t>(p);
    readAndMove<uint8_t>(p, "TRT.Minor");
    readAndMove<uint32_t>(p);
    readAndMove<uint8_t>(p, "TRT.Patch");
    readAndMove<uint32_t>(p);
    readAndMove<uint8_t>(p, "TRT.Build");
    readAndMove<uint64_t>(p);
    readAndMove<uint32_t>(p, "HardwareCompatLevel");
    readAndMove<uint64_t>(p);
    readAndMove<uint32_t>(p);

    printf("================================================================ Device information\n");
    CudaInfo eci; // Engine CUDA information
    eci.major = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);
    eci.minor = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);
    eci.maxCoreClockRate = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);
    eci.maxMemoryClockRate = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);
    eci.memoryBusWidth = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);
    eci.l2CacheSize = readAndMove<uint32_t>(p);
    readAndMove<uint64_t>(p);
    eci.maxPersistentL2CacheSize = readAndMove<uint32_t>(p);
    readAndMove<uint64_t>(p);
    eci.sharedMemPerBlock = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);
    eci.sharedMemPerMultiprocessor = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);
    eci.textureAlignment = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);
    eci.multiProcessorCount = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);
    eci.integrated = readAndMove<uint8_t>(p);
    readAndMove<uint32_t>(p);
    eci.maxThreadsPerBlock = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);
    eci.maxGridDimX = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);
    eci.maxGridDimY = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);
    eci.maxGridDimZ = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);
    if (trtMajor >= 10)
    {
        readAndMove<uint64_t>(p);
    }
    eci.totalGlobalMem = readAndMove<uint64_t>(p);
    readAndMove<uint32_t>(p);
    eci.maxTexture1DLinear = readAndMove<uint32_t>(p);
    readAndMove<uint32_t>(p);

    CudaInfo rci; // Runtime CUDA information
    cudaDeviceGetAttribute(&rci.major, cudaDevAttrComputeCapabilityMajor, iDevice);
    cudaDeviceGetAttribute(&rci.minor, cudaDevAttrComputeCapabilityMinor, iDevice);
    cudaDeviceGetAttribute(&rci.maxCoreClockRate, cudaDevAttrClockRate, iDevice);
    cudaDeviceGetAttribute(&rci.maxMemoryClockRate, cudaDevAttrMemoryClockRate, iDevice);
    cudaDeviceGetAttribute(&rci.memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, iDevice);
    int l2CacheSize = 0;
    cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, iDevice);
    rci.l2CacheSize              = l2CacheSize;
    int maxPersistentL2CacheSize = 0;
    cudaDeviceGetAttribute(&maxPersistentL2CacheSize, cudaDevAttrMaxPersistingL2CacheSize, iDevice);
    rci.maxPersistentL2CacheSize = maxPersistentL2CacheSize;
    cudaDeviceGetAttribute(&rci.sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, iDevice);
    cudaDeviceGetAttribute(&rci.sharedMemPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerMultiprocessor, iDevice);
    cudaDeviceGetAttribute(&rci.textureAlignment, cudaDevAttrTextureAlignment, iDevice);
    cudaDeviceGetAttribute(&rci.multiProcessorCount, cudaDevAttrMultiProcessorCount, iDevice);
    int integrated = false;
    cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, iDevice);
    rci.integrated = integrated;
    cudaDeviceGetAttribute(&rci.maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, iDevice);
    cudaDeviceGetAttribute(&rci.maxGridDimX, cudaDevAttrMaxGridDimX, iDevice);
    cudaDeviceGetAttribute(&rci.maxGridDimY, cudaDevAttrMaxGridDimY, iDevice);
    cudaDeviceGetAttribute(&rci.maxGridDimZ, cudaDevAttrMaxGridDimZ, iDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, iDevice);
    rci.totalGlobalMem     = prop.totalGlobalMem;
    rci.maxTexture1DLinear = prop.maxTexture1DLinear;

    printf("Property name               :     Engine      <->     Current     \n");
    PRINT(major);
    PRINT(minor);
    PRINT(maxCoreClockRate);
    PRINT(maxMemoryClockRate);
    PRINT(memoryBusWidth);
    PRINT(l2CacheSize);
    PRINT(maxPersistentL2CacheSize);
    PRINT(sharedMemPerBlock);
    PRINT(sharedMemPerMultiprocessor);
    PRINT(textureAlignment);
    PRINT(multiProcessorCount);
    PRINT(integrated);
    PRINT(maxThreadsPerBlock);
    PRINT(maxGridDimX);
    PRINT(maxGridDimY);
    PRINT(maxGridDimZ);
    PRINT(totalGlobalMem);
    PRINT(maxTexture1DLinear);

    printf("\nFinish\n");
    return 0;
}
