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

#include "NCCLAllReducePlugin.h"
#include "cookbookHelper.cuh"

#include <cmath>
#include <sstream>

using namespace nvinfer1;

const std::string pluginName {"NCCLAllReduce"};
const char       *inputTensorName {"inputT0"};
Dims64            shape {1, {4}};
static Logger     gLogger(ILogger::Severity::kERROR);

bool getEnvInt(char const *key, int32_t &value)
{
    char const *text = std::getenv(key);
    if (text == nullptr)
    {
        return false;
    }
    value = std::stoi(text);
    return true;
}

bool parseHexToBytes(std::string const &hex, char *out, size_t nByte)
{
    if (hex.size() != nByte * 2)
    {
        return false;
    }
    for (size_t i = 0; i < nByte; ++i)
    {
        std::string one = hex.substr(i * 2, 2);
        out[i]          = static_cast<char>(std::stoi(one, nullptr, 16));
    }
    return true;
}

bool runOneRank(int32_t rank, int32_t worldSize, std::string const &uidHex)
{
    CHECK(cudaSetDevice(0));

    ncclUniqueId uid {};
    if (!parseHexToBytes(uidHex, uid.internal, sizeof(uid.internal)))
    {
        std::cout << "Bad NCCL_UID_HEX" << std::endl;
        return false;
    }

    std::ostringstream oss;
    oss << "model-rank" << rank << ".trt";
    std::string trtFile = oss.str();

    IRuntime    *runtime {createInferRuntime(gLogger)};
    ICudaEngine *engine {nullptr};

    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream modelFile(trtFile, std::ios::binary | std::ios::ate);
        if (!modelFile)
        {
            std::cout << "Failed opening engine file for reading" << std::endl;
            return false;
        }
        std::streamsize modelSize = modelFile.tellg();
        modelFile.seekg(0, std::ios::beg);
        std::vector<char> modelData(modelSize);
        if (!modelFile.read(modelData.data(), modelSize))
        {
            std::cout << "Failed reading engine file" << std::endl;
            return false;
        }
        engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());
    }
    else
    {
        IBuilder           *builder = createInferBuilder(gLogger);
        INetworkDefinition *network = builder->createNetworkV2(0);
        IBuilderConfig     *config  = builder->createBuilderConfig();

        ITensor *inputTensor = network->addInput(inputTensorName, DataType::kFLOAT, shape);

        std::vector<PluginField> vecPF {
            {"rank", &rank, PluginFieldType::kINT32, 1},
            {"world_size", &worldSize, PluginFieldType::kINT32, 1},
            {"nccl_uid", uid.internal, PluginFieldType::kCHAR, static_cast<int32_t>(sizeof(uid.internal))},
        };
        PluginFieldCollection      pfc {static_cast<int32_t>(vecPF.size()), vecPF.data()};
        IPluginCreatorV3One       *pluginCreator {static_cast<IPluginCreatorV3One *>(getPluginRegistry()->getCreator(pluginName.c_str(), "1", ""))};
        std::unique_ptr<IPluginV3> plugin {pluginCreator->createPlugin(pluginName.c_str(), &pfc, TensorRTPhase::kBUILD)};

        std::vector<ITensor *> inputsVec {inputTensor};
        IPluginV3Layer        *pluginV3Layer = network->addPluginV3(inputsVec.data(), inputsVec.size(), nullptr, 0, *plugin);

        network->markOutput(*pluginV3Layer->getOutput(0));
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString == nullptr || engineString->size() == 0)
        {
            std::cout << "Fail building engine" << std::endl;
            return false;
        }

        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile)
        {
            std::cout << "Failed opening file to write" << std::endl;
            return false;
        }
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        if (engineFile.fail())
        {
            std::cout << "Fail saving engine" << std::endl;
            return false;
        }

        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
    }

    IExecutionContext *context = engine->createExecutionContext();
    context->setInputShape(inputTensorName, shape);

    if (std::getenv("NCCL_PREPARE_ONLY") != nullptr)
    {
        std::cout << "[rank " << rank << "] engine prepared" << std::endl;
        return true;
    }

    int const                 nIO = engine->getNbIOTensors();
    std::vector<const char *> tensorNameList(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        tensorNameList[i] = engine->getIOTensorName(i);
    }

    std::map<std::string, std::tuple<void *, void *, int>> bufferMap;
    for (auto const name : tensorNameList)
    {
        Dims64 dim {context->getTensorShape(name)};
        int    nByte        = std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<>()) * dataTypeToSize(engine->getTensorDataType(name));
        void  *hostBuffer   = (void *)new char[nByte];
        void  *deviceBuffer = nullptr;
        CHECK(cudaMalloc(&deviceBuffer, nByte));
        bufferMap[name] = std::make_tuple(hostBuffer, deviceBuffer, nByte);
    }

    float *pInputData = static_cast<float *>(std::get<0>(bufferMap[inputTensorName]));
    for (int i = 0; i < std::get<2>(bufferMap[inputTensorName]) / sizeof(float); ++i)
    {
        pInputData[i] = float(rank + 1);
    }

    for (auto const name : tensorNameList)
    {
        context->setTensorAddress(name, std::get<1>(bufferMap[name]));
    }

    for (auto const name : tensorNameList)
    {
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT)
        {
            CHECK(cudaMemcpy(std::get<1>(bufferMap[name]), std::get<0>(bufferMap[name]), std::get<2>(bufferMap[name]), cudaMemcpyHostToDevice));
        }
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    context->enqueueV3(stream);
    CHECK(cudaStreamSynchronize(stream));

    for (auto const name : tensorNameList)
    {
        if (engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT)
        {
            CHECK(cudaMemcpy(std::get<0>(bufferMap[name]), std::get<1>(bufferMap[name]), std::get<2>(bufferMap[name]), cudaMemcpyDeviceToHost));
        }
    }

    std::string outputName = tensorNameList[1];
    float      *out        = static_cast<float *>(std::get<0>(bufferMap[outputName]));
    float       expected   = float(worldSize * (worldSize + 1) / 2);
    bool        pass       = true;
    for (int i = 0; i < 4; ++i)
    {
        if (std::abs(out[i] - expected) > 1e-5f)
        {
            pass = false;
        }
    }

    std::cout << "[rank " << rank << "] output=";
    for (int i = 0; i < 4; ++i)
    {
        std::cout << out[i] << (i + 1 == 4 ? "" : ",");
    }
    std::cout << ", expected=" << expected << ", " << (pass ? "PASS" : "FAIL") << std::endl;

    for (auto const name : tensorNameList)
    {
        delete[] static_cast<char *>(std::get<0>(bufferMap[name]));
        CHECK(cudaFree(std::get<1>(bufferMap[name])));
    }
    CHECK(cudaStreamDestroy(stream));

    return pass;
}

int main()
{
    int32_t rank {0};
    int32_t worldSize {2};

    if (!getEnvInt("NCCL_RANK", rank) || !getEnvInt("NCCL_WORLD_SIZE", worldSize))
    {
        std::cout << "Please set NCCL_RANK and NCCL_WORLD_SIZE" << std::endl;
        return 1;
    }

    char const *uidHexCStr = std::getenv("NCCL_UID_HEX");
    if (uidHexCStr == nullptr)
    {
        std::cout << "Please set NCCL_UID_HEX" << std::endl;
        return 1;
    }

    bool pass = runOneRank(rank, worldSize, std::string(uidHexCStr));
    return pass ? 0 : 1;
}
