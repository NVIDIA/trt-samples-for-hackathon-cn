/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
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

// Two C++-side helpers from cookbookHelper, exercised on one small network:
//   + `DebugTensorWriter` - an IDebugListener that saves each debug tensor as a .npy file, so the
//     intermediate values can be diffed against a reference run in numpy afterwards. The Python
//     `CookbookDebugListener` only prints them.
//   + `launchGlobalTimerKernel` - reads the PTX %globaltimer register, the documented substitute
//     for cudaEventElapsedTime() under Confidential Compute.

#include "cookbookHelper.cuh"

using namespace nvinfer1;

// An IDebugListener that saves each debug tensor as a .npy file (through the bundled cnpy) instead
// of only printing it, so the values can be diffed against a reference run in numpy afterwards.
//
// This lives in the example rather than in cookbookHelper on purpose: it pulls in cnpy and zlib,
// and 30 other C++ examples link cookbookHelper without either of them.
class DebugTensorWriter : public IDebugListener
{
public:
    DebugTensorWriter(std::string prefix = "debug-tensor"):
        mPrefix(std::move(prefix)) {}

    bool processDebugTensor(void const *addr, TensorLocation location, DataType type, Dims const &shape, char const *name, cudaStream_t stream) noexcept override
    {
        int64_t nElement = std::accumulate(shape.d, shape.d + shape.nbDims, 1LL, std::multiplies<int64_t>());
        size_t  nByte    = nElement * dataTypeToSize(type);

        std::vector<char> hostBuffer(nByte);
        if (location == TensorLocation::kDEVICE)
        {
            CHECK(cudaStreamSynchronize(stream));
            CHECK(cudaMemcpyAsync(hostBuffer.data(), addr, nByte, cudaMemcpyDeviceToHost, stream));
            CHECK(cudaStreamSynchronize(stream));
        }
        else
        {
            std::memcpy(hostBuffer.data(), addr, nByte);
        }

        std::vector<size_t> cnpyShape;
        for (int i = 0; i < shape.nbDims; ++i)
        {
            cnpyShape.push_back(static_cast<size_t>(shape.d[i]));
        }

        // A tensor name can contain characters that are awkward in a path.
        std::string safeName(name);
        for (char &c : safeName)
        {
            if (c == '/' || c == '\\' || c == ' ' || c == ':')
            {
                c = '_';
            }
        }
        std::string fileName = mPrefix + "-" + safeName + ".npy";

        // cnpy needs the element type at compile time, so dispatch on the runtime DataType.
        switch (type)
        {
        case DataType::kFLOAT:
            cnpy::npy_save(fileName, reinterpret_cast<float const *>(hostBuffer.data()), cnpyShape, "w");
            break;
        case DataType::kINT32:
            cnpy::npy_save(fileName, reinterpret_cast<int32_t const *>(hostBuffer.data()), cnpyShape, "w");
            break;
        case DataType::kINT64:
            cnpy::npy_save(fileName, reinterpret_cast<int64_t const *>(hostBuffer.data()), cnpyShape, "w");
            break;
        default:
            // Types numpy has no direct equivalent for (FP16/BF16/FP8/INT4 ...) are written as raw
            // bytes, so they can still be reinterpreted on the numpy side.
            cnpy::npy_save(fileName, reinterpret_cast<uint8_t const *>(hostBuffer.data()), std::vector<size_t> {nByte}, "w");
            break;
        }

        std::cout << "[DebugTensorWriter] " << name << ", " << dataTypeToString(type) << ", "
                  << shapeToString(const_cast<Dims64 &>(shape)) << ", " << nByte << " bytes -> " << fileName << std::endl;
        return true;
    }

private:
    std::string mPrefix;
};

char const   *inputTensorName {"inputT0"};
char const   *debugTensorName {"a_cute_tensor"};
Dims64        shape {3, {3, 4, 5}};
static Logger gLogger(ILogger::Severity::kERROR);

int main()
{
    CHECK(cudaSetDevice(0));

    IBuilder           *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetworkV2(0);
    IBuilderConfig     *config  = builder->createBuilderConfig();

    ITensor           *inputTensor = network->addInput(inputTensorName, DataType::kFLOAT, shape);
    IElementWiseLayer *layer1      = network->addElementWise(*inputTensor, *inputTensor, ElementWiseOperation::kSUM);
    ITensor           *tensor1     = layer1->getOutput(0);
    tensor1->setName(debugTensorName);
    // Marking a tensor as debuggable keeps it from being fused away, so it stays observable.
    network->markDebug(*tensor1);
    IElementWiseLayer *layer2 = network->addElementWise(*tensor1, *tensor1, ElementWiseOperation::kSUM);
    network->markOutput(*layer2->getOutput(0));

    IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
    if (engineString == nullptr || engineString->size() == 0)
    {
        std::cout << "Fail building engine" << std::endl;
        return 1;
    }

    IRuntime          *runtime = createInferRuntime(gLogger);
    ICudaEngine       *engine  = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
    IExecutionContext *context = engine->createExecutionContext();

    // Route debug tensors into .npy files instead of just printing them.
    DebugTensorWriter debugTensorWriter("debug-tensor");
    context->setDebugListener(&debugTensorWriter);
    context->setTensorDebugState(debugTensorName, true);

    int const                                        nIO = engine->getNbIOTensors();
    std::map<std::string, std::pair<void *, void *>> bufferMap; // name -> (host, device)
    for (int i = 0; i < nIO; ++i)
    {
        char const *name         = engine->getIOTensorName(i);
        Dims64      dim          = context->getTensorShape(name);
        int         nByte        = std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<>()) * dataTypeToSize(engine->getTensorDataType(name));
        void       *hostBuffer   = new char[nByte];
        void       *deviceBuffer = nullptr;
        CHECK(cudaMalloc(&deviceBuffer, nByte));
        bufferMap[name] = std::make_pair(hostBuffer, deviceBuffer);
        context->setTensorAddress(name, deviceBuffer);
    }

    float *pInput = static_cast<float *>(bufferMap[inputTensorName].first);
    for (int i = 0; i < shape.d[0] * shape.d[1] * shape.d[2]; ++i)
    {
        pInput[i] = float(i);
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    int const nByteInput = shape.d[0] * shape.d[1] * shape.d[2] * sizeof(float);
    CHECK(cudaMemcpyAsync(bufferMap[inputTensorName].second, pInput, nByteInput, cudaMemcpyHostToDevice, stream));

    // Time the inference twice over: once with CUDA events, once with the GPU global timer. The two
    // should agree closely; under Confidential Compute only the second one stays trustworthy.
    uint64_t *dTimestamp = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&dTimestamp), 2 * sizeof(uint64_t)));

    cudaEvent_t eventStart, eventEnd;
    CHECK(cudaEventCreate(&eventStart));
    CHECK(cudaEventCreate(&eventEnd));

    CHECK(cudaEventRecord(eventStart, stream));
    CHECK(launchGlobalTimerKernel(dTimestamp, stream));

    context->enqueueV3(stream);

    CHECK(launchGlobalTimerKernel(dTimestamp + 1, stream));
    CHECK(cudaEventRecord(eventEnd, stream));
    CHECK(cudaStreamSynchronize(stream));

    float eventMs {0.0F};
    CHECK(cudaEventElapsedTime(&eventMs, eventStart, eventEnd));

    uint64_t hTimestamp[2] {};
    CHECK(cudaMemcpy(hTimestamp, dTimestamp, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    double globalTimerMs = static_cast<double>(hTimestamp[1] - hTimestamp[0]) / 1.0e6;

    std::cout << "cudaEventElapsedTime : " << eventMs << " ms" << std::endl;
    std::cout << "%globaltimer         : " << globalTimerMs << " ms" << std::endl;

    CHECK(cudaEventDestroy(eventStart));
    CHECK(cudaEventDestroy(eventEnd));
    CHECK(cudaFree(dTimestamp));
    CHECK(cudaStreamDestroy(stream));

    for (auto const &[name, buffers] : bufferMap)
    {
        delete[] static_cast<char *>(buffers.first);
        CHECK(cudaFree(buffers.second));
    }
    delete context;
    delete engine;
    delete runtime;
    delete engineString;
    delete config;
    delete network;
    delete builder;

    std::cout << "Finish" << std::endl;
    return 0;
}
