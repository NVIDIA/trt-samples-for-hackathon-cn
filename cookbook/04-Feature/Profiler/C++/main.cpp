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

//! Report latency and per-layer time from C++, the way trtexec does.
//!
//! `07-Tool/trtexec/parse_export_json.py` already computes percentiles -- but only *after*
//! trtexec has written its JSON. A C++ service that embeds TensorRT has no trtexec and no
//! JSON; it has to measure itself. `samples/common/sampleReporting.{h,cpp}` in TensorRT-OSS
//! is the code trtexec uses for this, and this file re-expresses the parts of it that a
//! deployed application actually needs, against the public API only.
//!
//! Three ideas are worth taking from it:
//!
//! 1. **A single end-to-end number hides where the time goes.** Time H2D, compute and D2H
//!    separately with their own CUDA events, and report the enqueue (host) cost too -- on this
//!    network that last one is the finding: enqueue is more than half of compute, so the
//!    engine is launch-bound and no kernel would fix it.
//! 2. **Report percentiles and the coefficient of variation, not the mean.** The mean of a
//!    latency distribution with a tail is not a latency anyone experiences.
//! 3. **Aggregate the per-layer profile with the median, not the sum or the mean.** An
//!    `IProfiler` is called once per layer per execution; one slow first iteration would
//!    dominate a mean. This is exactly what `LayerProfile::median` in the upstream file does.

#include "cookbookHelper.cuh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <numeric>

using namespace nvinfer1;

static Logger gLogger(ILogger::Severity::kERROR);

int32_t const kWarmup {20};
int32_t const kIteration {200};
int32_t const kBatch {4}; // The largest batch `buildMnistNetwork`'s optimization profile allows

//! Per-iteration measurement, mirroring `InferenceTime` in the upstream file.
struct InferenceTime
{
    float enqueue {0.0F}; //!< Host time spent inside enqueueV3, i.e. launch cost
    float h2d {0.0F};
    float compute {0.0F};
    float d2h {0.0F};

    float latency() const
    {
        return h2d + compute + d2h;
    }
};

//! Summary of one metric over all iterations, mirroring `PerformanceResult`.
struct PerformanceResult
{
    float              min {0.0F};
    float              max {0.0F};
    float              mean {0.0F};
    float              median {0.0F};
    std::vector<float> percentile; // p90, p95, p99
    float              coeffVar {0.0F};
};

//! Linear-interpolation percentile, the same convention trtexec reports.
float percentileOf(std::vector<float> sorted, float p)
{
    if (sorted.empty())
    {
        return 0.0F;
    }
    float const rank  = (p / 100.0F) * static_cast<float>(sorted.size() - 1);
    size_t      lower = static_cast<size_t>(std::floor(rank));
    size_t      upper = static_cast<size_t>(std::ceil(rank));
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - static_cast<float>(lower));
}

PerformanceResult summarize(std::vector<float> value)
{
    PerformanceResult result {};
    if (value.empty())
    {
        return result;
    }
    std::sort(value.begin(), value.end());
    result.min    = value.front();
    result.max    = value.back();
    result.mean   = std::accumulate(value.begin(), value.end(), 0.0F) / static_cast<float>(value.size());
    result.median = percentileOf(value, 50.0F);
    for (float p : {90.0F, 95.0F, 99.0F})
    {
        result.percentile.push_back(percentileOf(value, p));
    }
    // Coefficient of variation: standard deviation as a fraction of the mean. A run with a
    // low coefficient of variation is one whose numbers are worth quoting at all.
    float accumulator {0.0F};
    for (float v : value)
    {
        accumulator += (v - result.mean) * (v - result.mean);
    }
    float const stddev = std::sqrt(accumulator / static_cast<float>(value.size()));
    result.coeffVar    = result.mean > 0.0F ? stddev / result.mean * 100.0F : 0.0F;
    return result;
}

//! Collects one entry per layer per execution, then reduces with the median.
class MedianProfiler : public IProfiler
{
public:
    void reportLayerTime(char const *layerName, float ms) noexcept override
    {
        mRecord[layerName].push_back(ms);
    }

    //! (name, median ms) sorted by descending median.
    std::vector<std::pair<std::string, float>> medianPerLayer() const
    {
        std::vector<std::pair<std::string, float>> out;
        for (auto const &entry : mRecord)
        {
            std::vector<float> value {entry.second};
            std::sort(value.begin(), value.end());
            out.emplace_back(entry.first, percentileOf(value, 50.0F));
        }
        std::sort(out.begin(), out.end(), [](auto const &a, auto const &b)
                  { return a.second > b.second; });
        return out;
    }

    size_t iterationCount() const
    {
        return mRecord.empty() ? 0 : mRecord.begin()->second.size();
    }

private:
    std::map<std::string, std::vector<float>> mRecord;
};

void printResult(std::string const &name, PerformanceResult const &result, std::string const &unit)
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(12) << name << " | " << std::setw(9) << result.min << std::setw(10) << result.max << std::setw(10) << result.mean << std::setw(10) << result.median;
    for (float p : result.percentile)
    {
        std::cout << std::setw(10) << p;
    }
    std::cout << std::setw(9) << std::setprecision(2) << result.coeffVar << "%  " << unit << std::endl;
}

int main()
{
    // ---- The cookbook's MNIST network, the same one `../main.py` measures
    IBuilder           *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
    IBuilderConfig     *config  = builder->createBuilderConfig();

    IOptimizationProfile *profile = builder->createOptimizationProfile();
    for (ITensor *tensor : buildMnistNetwork(config, network, profile))
    {
        network->markOutput(*tensor);
    }

    IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
    if (engineString == nullptr || engineString->size() == 0)
    {
        std::cout << "Failed building engine" << std::endl;
        return 1;
    }
    IRuntime          *runtime {createInferRuntime(gLogger)};
    ICudaEngine       *engine {runtime->deserializeCudaEngine(engineString->data(), engineString->size())};
    IExecutionContext *context {engine->createExecutionContext()};

    // ---- Buffers
    int32_t const            nIO = engine->getNbIOTensors();
    std::vector<std::string> nameList(nIO);
    std::vector<void *>      deviceBuffer(nIO, nullptr);
    std::vector<void *>      hostBuffer(nIO, nullptr);
    std::vector<size_t>      byteList(nIO, 0);
    for (int32_t i = 0; i < nIO; ++i) // The network has a dynamic batch, so a shape must be chosen first
    {
        nameList[i] = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(nameList[i].c_str()) == TensorIOMode::kINPUT)
        {
            Dims64 shape = engine->getTensorShape(nameList[i].c_str());
            shape.d[0]   = kBatch;
            context->setInputShape(nameList[i].c_str(), shape);
        }
    }
    for (int32_t i = 0; i < nIO; ++i)
    {
        Dims64 shape = context->getTensorShape(nameList[i].c_str());
        size_t count = 1;
        for (int32_t j = 0; j < shape.nbDims; ++j)
        {
            count *= shape.d[j];
        }
        // Not `sizeof(float)` for every tensor: the MNIST network's `z` output is int64
        byteList[i] = count * dataTypeToSize(engine->getTensorDataType(nameList[i].c_str()));
        CHECK(cudaMalloc(&deviceBuffer[i], byteList[i]));
        CHECK(cudaMallocHost(&hostBuffer[i], byteList[i])); // Pinned, or H2D/D2H measure the copy engine badly
        std::fill_n(static_cast<char *>(hostBuffer[i]), byteList[i], 0);
        context->setTensorAddress(nameList[i].c_str(), deviceBuffer[i]);
    }

    cudaStream_t stream {nullptr};
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t h2dStart, h2dEnd, computeStart, computeEnd, d2hStart, d2hEnd;
    for (cudaEvent_t *e : {&h2dStart, &h2dEnd, &computeStart, &computeEnd, &d2hStart, &d2hEnd})
    {
        CHECK(cudaEventCreate(e));
    }

    auto runOnce = [&](InferenceTime &time)
    {
        CHECK(cudaEventRecord(h2dStart, stream));
        for (int32_t i = 0; i < nIO; ++i)
        {
            if (engine->getTensorIOMode(nameList[i].c_str()) == TensorIOMode::kINPUT)
            {
                CHECK(cudaMemcpyAsync(deviceBuffer[i], hostBuffer[i], byteList[i], cudaMemcpyHostToDevice, stream));
            }
        }
        CHECK(cudaEventRecord(h2dEnd, stream));

        CHECK(cudaEventRecord(computeStart, stream));
        auto const enqueueStart = std::chrono::high_resolution_clock::now();
        context->enqueueV3(stream);
        auto const enqueueEnd = std::chrono::high_resolution_clock::now();
        CHECK(cudaEventRecord(computeEnd, stream));

        CHECK(cudaEventRecord(d2hStart, stream));
        for (int32_t i = 0; i < nIO; ++i)
        {
            if (engine->getTensorIOMode(nameList[i].c_str()) == TensorIOMode::kOUTPUT)
            {
                CHECK(cudaMemcpyAsync(hostBuffer[i], deviceBuffer[i], byteList[i], cudaMemcpyDeviceToHost, stream));
            }
        }
        CHECK(cudaEventRecord(d2hEnd, stream));
        CHECK(cudaStreamSynchronize(stream));

        CHECK(cudaEventElapsedTime(&time.h2d, h2dStart, h2dEnd));
        CHECK(cudaEventElapsedTime(&time.compute, computeStart, computeEnd));
        CHECK(cudaEventElapsedTime(&time.d2h, d2hStart, d2hEnd));
        time.enqueue = std::chrono::duration<float, std::milli>(enqueueEnd - enqueueStart).count();
    };

    // ---- Warm up, then measure
    InferenceTime scratch {};
    for (int32_t i = 0; i < kWarmup; ++i)
    {
        runOnce(scratch);
    }

    std::vector<InferenceTime> timing(kIteration);
    for (int32_t i = 0; i < kIteration; ++i)
    {
        runOnce(timing[i]);
    }

    auto extract = [&](float InferenceTime::*member)
    {
        std::vector<float> out;
        out.reserve(timing.size());
        for (auto const &t : timing)
        {
            out.push_back(t.*member);
        }
        return out;
    };
    std::vector<float> latency;
    latency.reserve(timing.size());
    for (auto const &t : timing)
    {
        latency.push_back(t.latency());
    }

    std::cout << "=== Latency over " << kIteration << " iterations (" << kWarmup << " warm-up), batch " << kBatch << std::endl;
    std::cout << std::setw(12) << "metric"
              << " | " << std::setw(9) << "min" << std::setw(10) << "max" << std::setw(10) << "mean" << std::setw(10) << "median" << std::setw(10) << "p90" << std::setw(10) << "p95" << std::setw(10) << "p99" << std::setw(9) << "cv" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    printResult("H2D", summarize(extract(&InferenceTime::h2d)), "ms");
    printResult("compute", summarize(extract(&InferenceTime::compute)), "ms");
    printResult("D2H", summarize(extract(&InferenceTime::d2h)), "ms");
    printResult("latency", summarize(latency), "ms  (H2D + compute + D2H)");
    printResult("enqueue", summarize(extract(&InferenceTime::enqueue)), "ms  (host, launch cost)");

    PerformanceResult const latencyResult = summarize(latency);
    std::cout << std::setprecision(4) << "\nThroughput: " << 1000.0F / latencyResult.median * kBatch << " inference/s at the median" << std::endl;
    std::cout << "p99 / median = " << std::setprecision(2) << latencyResult.percentile[2] / latencyResult.median << "x -- the tail the mean would have hidden" << std::endl;

    // ---- Per-layer profile, reduced with the median
    MedianProfiler profiler;
    context->setProfiler(&profiler);
    for (int32_t i = 0; i < 50; ++i)
    {
        runOnce(scratch);
    }
    // NOTE: there is no way to *detach* a profiler. `setProfiler(nullptr)` is rejected by a
    // parameter check --
    //     Error Code 3: API Usage Error (Parameter check failed, condition: (profiler) != nullptr)
    // -- so profiling is a one-way switch for the lifetime of the context. Profile in a context
    // you are about to destroy, or keep a separate un-profiled context for the hot path.

    auto const perLayer = profiler.medianPerLayer();
    float      total {0.0F};
    for (auto const &entry : perLayer)
    {
        total += entry.second;
    }
    std::cout << "\n=== Per-layer time, median over " << profiler.iterationCount() << " executions" << std::endl;
    std::cout << std::setw(10) << "median ms" << std::setw(10) << "share"
              << "  layer" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    for (auto const &entry : perLayer)
    {
        std::cout << std::setw(10) << std::setprecision(4) << entry.second << std::setw(9) << std::setprecision(1) << entry.second / total * 100.0F << "%  " << entry.first << std::endl;
    }
    std::cout << std::string(100, '-') << std::endl;
    std::cout << std::setw(10) << std::setprecision(4) << total << "           (sum of per-layer medians)" << std::endl;
    std::cout << "Profiled compute median " << std::setprecision(4) << summarize(extract(&InferenceTime::compute)).median << " ms without the profiler attached." << std::endl;
    std::cout << "The profiler serialises layers to time them, so the sum above is expected to be larger." << std::endl;

    // ---- Release, in the reverse order of creation
    for (cudaEvent_t e : {h2dStart, h2dEnd, computeStart, computeEnd, d2hStart, d2hEnd})
    {
        CHECK(cudaEventDestroy(e));
    }
    CHECK(cudaStreamDestroy(stream));
    for (int32_t i = 0; i < nIO; ++i)
    {
        CHECK(cudaFree(deviceBuffer[i]));
        CHECK(cudaFreeHost(hostBuffer[i]));
    }
    delete context;
    delete engine;
    delete runtime;
    delete engineString;
    delete config;
    delete network;
    delete builder;

    std::cout << "\nFinish" << std::endl;
    return 0;
}
