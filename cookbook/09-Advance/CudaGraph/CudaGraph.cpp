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

void run()
{
    ICudaEngine *engine = nullptr;

    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0)
        {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr)
        {
            std::cout << "Failed loading engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        IBuilder *            builder = createInferBuilder(gLogger);
        INetworkDefinition *  network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        IBuilderConfig *      config  = builder->createBuilderConfig();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);

        ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims32 {3, {-1, -1, -1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {3, {3, 4, 5}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {3, {6, 8, 10}});
        config->addOptimizationProfile(profile);

        IIdentityLayer *identityLayer = network->addIdentity(*inputTensor);
        network->markOutput(*identityLayer->getOutput(0));
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString == nullptr || engineString->size() == 0)
        {
            std::cout << "Failed building serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr)
        {
            std::cout << "Failed building engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building engine!" << std::endl;

        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile)
        {
            std::cout << "Failed opening file to write" << std::endl;
            return;
        }
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        if (engineFile.fail())
        {
            std::cout << "Failed saving .plan file!" << std::endl;
            return;
        }
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }

    IExecutionContext *context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims32 {3, {3, 4, 5}});
    std::cout << std::string("Binding all? ") << std::string(context->allInputDimensionsSpecified() ? "Yes" : "No") << std::endl;
    int nBinding = engine->getNbBindings();
    int nInput   = 0;
    for (int i = 0; i < nBinding; ++i)
    {
        nInput += int(engine->bindingIsInput(i));
    }
    int nOutput = nBinding - nInput;
    for (int i = 0; i < nBinding; ++i)
    {
        std::cout << std::string("Bind[") << i << std::string(i < nInput ? "]:i[" : "]:o[") << (i < nInput ? i : i - nInput) << std::string("]->");
        std::cout << dataTypeToString(engine->getBindingDataType(i)) << std::string(" ");
        std::cout << shapeToString(context->getBindingDimensions(i)) << std::string(" ");
        std::cout << engine->getBindingName(i) << std::endl;
    }

    std::vector<int> vBindingSize(nBinding, 0);
    for (int i = 0; i < nBinding; ++i)
    {
        Dims32 dim  = context->getBindingDimensions(i);
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j)
        {
            size *= dim.d[j];
        }
        vBindingSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
    }

    std::vector<void *> vBufferH {nBinding, nullptr};
    std::vector<void *> vBufferD {nBinding, nullptr};
    for (int i = 0; i < nBinding; ++i)
    {
        vBufferH[i] = (void *)new char[vBindingSize[i]];
        CHECK(cudaMalloc(&vBufferD[i], vBindingSize[i]));
    }

    float *pData = (float *)vBufferH[0];
    for (int i = 0; i < vBindingSize[0] / dataTypeToSize(engine->getBindingDataType(0)); ++i)
    {
        pData[i] = float(i);
    }

    int  inputSize = 3 * 4 * 5, outputSize = 1;
    Dims outputShape = context->getBindingDimensions(1);
    for (int i = 0; i < outputShape.nbDims; ++i)
    {
        outputSize *= outputShape.d[i];
    }
    std::vector<float>  inputH0(inputSize, 1.0f);
    std::vector<float>  outputH0(outputSize, 0.0f);
    std::vector<void *> binding = {nullptr, nullptr};
    CHECK(cudaMalloc(&binding[0], sizeof(float) * inputSize));
    CHECK(cudaMalloc(&binding[1], sizeof(float) * outputSize));
    for (int i = 0; i < inputSize; ++i)
    {
        inputH0[i] = (float)i;
    }

    // 运行推理和使用 CUDA Graph 要用的流
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // 捕获 CUDA Graph 之前要运行一次推理，https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_execution_context.html#a2f4429652736e8ef6e19f433400108c7
    for (int i = 0; i < nInput; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice, stream));
    }

    context->enqueueV2(vBufferD.data(), stream, nullptr);

    for (int i = nInput; i < nBinding; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream); // 不用在 graph 内同步

    for (int i = 0; i < nBinding; ++i)
    {
        printArrayInfomation((float *)vBufferH[i], context->getBindingDimensions(i), std::string(engine->getBindingName(i)), true);
    }

    // 首次捕获 CUDA Graph 并运行推理
    cudaGraph_t     graph;
    cudaGraphExec_t graphExec = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < nInput; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice, stream));
    }

    context->enqueueV2(vBufferD.data(), stream, nullptr);

    for (int i = nInput; i < nBinding; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost, stream));
    }
    //cudaStreamSynchronize(stream); // 不用在 graph 内同步
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    // 输入尺寸改变后，也需要首先运行一次推理，然后重新捕获 CUDA Graph，最后再运行推理
    context->setBindingDimensions(0, Dims32 {3, {2, 3, 4}});
    std::cout << std::string("Binding all? ") << std::string(context->allInputDimensionsSpecified() ? "Yes" : "No") << std::endl;

    for (int i = 0; i < nBinding; ++i)
    {
        Dims32 dim  = context->getBindingDimensions(i);
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j)
        {
            size *= dim.d[j];
        }
        vBindingSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
    }

    // 这里偷懒，因为本次推理绑定的输入输出数据形状不大于上一次推理，所以这里不再重新准备所有 buffer

    for (int i = 0; i < nInput; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice, stream));
    }

    context->enqueueV2(vBufferD.data(), stream, nullptr);

    for (int i = nInput; i < nBinding; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream); // 不用在 graph 内同步

    for (int i = 0; i < nBinding; ++i)
    {
        printArrayInfomation((float *)vBufferH[i], context->getBindingDimensions(i), std::string(engine->getBindingName(i)), true);
    }

    // 再次捕获 CUDA Graph 并运行推理
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < nInput; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice, stream));
    }

    context->enqueueV2(vBufferD.data(), stream, nullptr);

    for (int i = nInput; i < nBinding; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost, stream));
    }
    //cudaStreamSynchronize(stream); // 不用在 graph 内同步
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);

    for (int i = 0; i < nBinding; ++i)
    {
        delete[] vBufferH[i];
        CHECK(cudaFree(vBufferD[i]));
    }
    return;
}

int main()
{
    CHECK(cudaSetDevice(0));
    run();
    run();
    return 0;
}
