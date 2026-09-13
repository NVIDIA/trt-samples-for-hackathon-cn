/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

//! Loading one engine onto N GPUs: sequentially, or all at once with `std::async`?
//!
//! `../main.py` shows that engine **bytes** are portable across devices while an `ICudaEngine`
//! is not, so a multi-GPU process has to deserialize the same plan once per device. On a
//! server with 8 GPUs and a multi-gigabyte plan, that startup cost is the difference between
//! a fast rollout and a slow one -- and the obvious optimisation is to do the N
//! deserializations concurrently instead of one after another.
//!
//! The obvious optimisation is not obviously correct, because it is not obvious what
//! deserialization is bound by. If it is dominated by host-side work (parsing the plan,
//! allocating, building kernel tables) threads help. If it is dominated by the H2D copy of
//! the weights, N threads share one PCIe root complex and help much less. This program
//! measures it rather than assuming.
//!
//! Rules the code has to respect, and each one is a real trap:
//!
//! 1. **`cudaSetDevice` is per-thread.** A thread spawned by `std::async` inherits nothing;
//!    it starts on device 0. Every worker must call `cudaSetDevice` itself, or all N engines
//!    land on the same GPU and the "parallel" number is meaningless *and still succeeds*.
//! 2. **One `IRuntime` per device.** A runtime carries device state; sharing one across
//!    threads that are on different devices is not what it is for.
//! 3. **The plan is read once, into a host buffer**, and shared read-only. Re-reading the file
//!    per device would measure the page cache instead of TensorRT.
//!
//! Re-expressed from the idea in the internal `samples_internal/deserializeTimer`; no code
//! was taken from it.

#include "cookbookHelper.cuh"

#include <chrono>
#include <future>
#include <numeric>
#include <thread>

using namespace nvinfer1;

static Logger gLogger(ILogger::Severity::kERROR);

int32_t const kRepeat {3};    // Repeat the whole experiment and take the best, to cut noise
int32_t const kChannel {256}; // Wide enough that the plan is large, so deserialization is measurable
int32_t const kSize {64};
int32_t const kNbConvolution {8};

//! Build one engine whose plan is big enough to time. The weights dominate the plan size.
std::vector<char> buildPlan()
{
    IBuilder           *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
    IBuilderConfig     *config  = builder->createBuilderConfig();

    ITensor *tensor = network->addInput("inputT0", DataType::kFLOAT, Dims64 {4, {1, kChannel, kSize, kSize}});

    int32_t const      weightCount = kChannel * kChannel * 3 * 3;
    std::vector<float> biasData(kChannel, 0.0F);
    // Every convolution needs its OWN weights. Giving them all the same buffer makes the plan
    // 2 MiB instead of 18 MiB, because TensorRT deduplicates identical weights in the engine --
    // and a 2 MiB plan deserializes too fast to measure anything. The `Weights` structs point
    // into `weightStore`, which must outlive `buildSerializedNetwork`.
    std::vector<std::vector<float>> weightStore(kNbConvolution, std::vector<float>(weightCount));
    for (int32_t i = 0; i < kNbConvolution; ++i)
    {
        for (int32_t j = 0; j < weightCount; ++j)
        {
            weightStore[i][j] = static_cast<float>((j * (i + 1) % 97) - 48) * 0.001F;
        }
    }
    for (int32_t i = 0; i < kNbConvolution; ++i)
    {
        Weights weight {DataType::kFLOAT, weightStore[i].data(), weightCount};
        Weights bias {DataType::kFLOAT, biasData.data(), kChannel};
        auto   *convolutionLayer = network->addConvolutionNd(*tensor, kChannel, Dims64 {2, {3, 3}}, weight, bias);
        convolutionLayer->setPaddingNd(Dims64 {2, {1, 1}});
        tensor = network->addActivation(*convolutionLayer->getOutput(0), ActivationType::kRELU)->getOutput(0);
    }
    network->markOutput(*tensor);

    IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
    if (engineString == nullptr || engineString->size() == 0)
    {
        std::cout << "Failed building engine" << std::endl;
        exit(1);
    }
    std::vector<char> plan(static_cast<char const *>(engineString->data()), static_cast<char const *>(engineString->data()) + engineString->size());

    delete engineString;
    delete config;
    delete network;
    delete builder;
    return plan;
}

//! Deserialize `plan` onto `device`. Returns the milliseconds this one took.
float deserializeOnDevice(std::vector<char> const &plan, int32_t device, ICudaEngine **outEngine, IRuntime **outRuntime)
{
    // Trap 1: a worker thread starts on device 0, whatever the parent was doing
    CHECK(cudaSetDevice(device));
    auto const start = std::chrono::high_resolution_clock::now();

    IRuntime    *runtime = createInferRuntime(gLogger); // Trap 2: one runtime per device
    ICudaEngine *engine  = runtime->deserializeCudaEngine(plan.data(), plan.size());

    auto const end = std::chrono::high_resolution_clock::now();
    if (engine == nullptr)
    {
        std::cout << "Failed deserializing on device " << device << std::endl;
        exit(1);
    }
    *outEngine  = engine;
    *outRuntime = runtime;
    return std::chrono::duration<float, std::milli>(end - start).count();
}

void release(std::vector<ICudaEngine *> &engineList, std::vector<IRuntime *> &runtimeList)
{
    for (size_t i = 0; i < engineList.size(); ++i)
    {
        cudaSetDevice(static_cast<int32_t>(i));
        delete engineList[i];
        delete runtimeList[i];
    }
    engineList.clear();
    runtimeList.clear();
}

int main()
{
    int32_t deviceCount {0};
    CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2)
    {
        std::cout << "Skip since no enough GPU is ready (need 2, get " << deviceCount << ")" << std::endl;
        return 0;
    }

    std::vector<char> const plan = buildPlan();
    std::cout << "Devices: " << deviceCount << ", plan size: " << plan.size() / (1 << 20) << " MiB" << std::endl;

    float              sequentialBest {1e30F}, parallelBest {1e30F};
    std::vector<float> lastSequentialPerDevice, lastParallelPerDevice;

    for (int32_t repeat = 0; repeat < kRepeat; ++repeat)
    {
        // ---- Sequential: one device after another, on this thread
        {
            std::vector<ICudaEngine *> engineList(deviceCount, nullptr);
            std::vector<IRuntime *>    runtimeList(deviceCount, nullptr);
            std::vector<float>         perDevice(deviceCount, 0.0F);

            auto const start = std::chrono::high_resolution_clock::now();
            for (int32_t device = 0; device < deviceCount; ++device)
            {
                perDevice[device] = deserializeOnDevice(plan, device, &engineList[device], &runtimeList[device]);
            }
            float const total       = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
            sequentialBest          = std::min(sequentialBest, total);
            lastSequentialPerDevice = perDevice;
            release(engineList, runtimeList);
        }

        // ---- Parallel: one std::async per device
        {
            std::vector<ICudaEngine *> engineList(deviceCount, nullptr);
            std::vector<IRuntime *>    runtimeList(deviceCount, nullptr);

            auto const                      start = std::chrono::high_resolution_clock::now();
            std::vector<std::future<float>> futureList;
            for (int32_t device = 0; device < deviceCount; ++device)
            {
                // std::launch::async forces a real thread; the default policy is allowed to
                // run lazily on get(), which would silently make this the sequential case.
                futureList.push_back(std::async(std::launch::async, [&plan, device, &engineList, &runtimeList]()
                                                { return deserializeOnDevice(plan, device, &engineList[device], &runtimeList[device]); }));
            }
            std::vector<float> perDevice;
            for (auto &future : futureList)
            {
                perDevice.push_back(future.get());
            }
            float const total     = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
            parallelBest          = std::min(parallelBest, total);
            lastParallelPerDevice = perDevice;
            release(engineList, runtimeList);
        }
    }

    auto sum = [](std::vector<float> const &v)
    { return std::accumulate(v.begin(), v.end(), 0.0F); };
    auto max = [](std::vector<float> const &v)
    { return *std::max_element(v.begin(), v.end()); };

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n=== Deserializing the same plan onto " << deviceCount << " GPUs (best of " << kRepeat << ")" << std::endl;
    std::cout << std::string(78, '-') << std::endl;
    std::cout << "sequential : wall " << std::setw(8) << sequentialBest << " ms   per-device " << sum(lastSequentialPerDevice) / deviceCount << " ms avg" << std::endl;
    std::cout << "parallel   : wall " << std::setw(8) << parallelBest << " ms   per-device " << sum(lastParallelPerDevice) / deviceCount << " ms avg, slowest " << max(lastParallelPerDevice) << " ms" << std::endl;
    std::cout << std::string(78, '-') << std::endl;
    std::cout << "speed-up   : " << std::setprecision(2) << sequentialBest / parallelBest << "x  (perfect scaling would be " << deviceCount << "x)" << std::endl;
    std::cout << "efficiency : " << std::setprecision(0) << (sequentialBest / parallelBest) / deviceCount * 100.0F << "% of linear" << std::endl;

    std::cout << "\nPer-device time, parallel run:";
    for (size_t i = 0; i < lastParallelPerDevice.size(); ++i)
    {
        std::cout << " [" << i << "]=" << std::setprecision(1) << lastParallelPerDevice[i] << "ms";
    }
    std::cout << "\nPer-device time, sequential run:";
    for (size_t i = 0; i < lastSequentialPerDevice.size(); ++i)
    {
        std::cout << " [" << i << "]=" << std::setprecision(1) << lastSequentialPerDevice[i] << "ms";
    }
    std::cout << std::endl;
    std::cout << "\nIf the parallel per-device times are much larger than the sequential ones, the" << std::endl;
    std::cout << "deserializations are contending for a shared resource rather than running freely." << std::endl;

    std::cout << "\nFinish" << std::endl;
    return 0;
}
