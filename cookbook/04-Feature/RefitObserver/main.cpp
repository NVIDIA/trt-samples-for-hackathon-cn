// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// IRefitterObserver: record at build time how every refittable engine weight is produced from the
// ONNX graph, then replay that recording at deploy time to refit without an ONNX parser.
//
// Added in TensorRT 11.2 (NvOnnxParser.h), C++ only -- there is no Python binding.

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace nvinfer1;

#define CHECK(call)                                                                \
    do                                                                             \
    {                                                                              \
        cudaError_t e = (call);                                                    \
        if (e != cudaSuccess)                                                      \
        {                                                                          \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
            return 1;                                                              \
        }                                                                          \
    } while (0)

class Logger : public ILogger
{
    void log(Severity severity, char const *msg) noexcept override
    {
        if (severity <= Severity::kERROR)
        {
            printf("    [TRT] %s\n", msg);
        }
    }
} gLogger;

char const *kindName(nvonnxparser::RefitTransformKind kind)
{
    switch (kind)
    {
    case nvonnxparser::RefitTransformKind::kIDENTITY: return "kIDENTITY";
    case nvonnxparser::RefitTransformKind::kDOUBLE_TO_FLOAT: return "kDOUBLE_TO_FLOAT";
    case nvonnxparser::RefitTransformKind::kBATCH_NORM_FOLD_SCALE: return "kBATCH_NORM_FOLD_SCALE";
    case nvonnxparser::RefitTransformKind::kBATCH_NORM_FOLD_BIAS: return "kBATCH_NORM_FOLD_BIAS";
    case nvonnxparser::RefitTransformKind::kCONSTANT_NODE: return "kCONSTANT_NODE";
    case nvonnxparser::RefitTransformKind::kCONSTANT_OF_SHAPE: return "kCONSTANT_OF_SHAPE";
    }
    return "unknown";
}

// ------------------------------------------------------------------------------------------------
// The recording
// ------------------------------------------------------------------------------------------------

struct Entry
{
    std::string              trtName;
    int32_t                  kind {0};
    int32_t                  trtDtype {0};
    int64_t                  count {0};
    float                    epsilon {0.0F};
    std::vector<std::string> sources;
    std::vector<uint8_t>     fixedData; // only for the kCONSTANT* kinds
};

// The observer copies everything it is given. RefitRecord's pointers are owned by the parser and
// are valid only for the duration of the call -- keeping one is a use-after-free.
class TableRecorder : public nvonnxparser::IRefitterObserver
{
public:
    void onRefittableWeight(nvonnxparser::RefitRecord const &record) noexcept override
    {
        Entry entry;
        entry.trtName  = record.trtName;
        entry.kind     = static_cast<int32_t>(record.kind);
        entry.trtDtype = static_cast<int32_t>(record.trtDtype);
        entry.count    = record.count;
        entry.epsilon  = record.epsilon;
        for (int32_t i = 0; i < record.nbSources; ++i)
        {
            entry.sources.emplace_back(record.sourceOnnxNames[i]);
        }
        if (record.fixedData != nullptr && record.fixedDataSize > 0)
        {
            auto const *bytes = static_cast<uint8_t const *>(record.fixedData);
            entry.fixedData.assign(bytes, bytes + record.fixedDataSize);
        }
        table.push_back(std::move(entry));
    }

    std::vector<Entry> table;
};

void writeTable(std::vector<Entry> const &table, std::string const &path)
{
    std::ofstream out(path);
    out << table.size() << "\n";
    for (auto const &e : table)
    {
        out << e.trtName << " " << e.kind << " " << e.trtDtype << " " << e.count << " "
            << std::scientific << std::setprecision(9) << e.epsilon << " " << e.sources.size();
        for (auto const &s : e.sources)
        {
            out << " " << s;
        }
        out << " " << e.fixedData.size();
        for (auto b : e.fixedData)
        {
            out << " " << static_cast<int32_t>(b);
        }
        out << "\n";
    }
}

std::vector<Entry> readTable(std::string const &path)
{
    std::ifstream in(path);
    size_t        n {0};
    in >> n;
    std::vector<Entry> table(n);
    for (auto &e : table)
    {
        size_t nSource {0}, nByte {0};
        in >> e.trtName >> e.kind >> e.trtDtype >> e.count >> e.epsilon >> nSource;
        e.sources.resize(nSource);
        for (auto &s : e.sources)
        {
            in >> s;
        }
        in >> nByte;
        e.fixedData.resize(nByte);
        for (auto &b : e.fixedData)
        {
            int32_t v {0};
            in >> v;
            b = static_cast<uint8_t>(v);
        }
    }
    return table;
}

// ------------------------------------------------------------------------------------------------
// The `.wts` reader, so the deploy step needs no protobuf. See 02-API/Network/weight_transport.py.
// ------------------------------------------------------------------------------------------------

struct Blob
{
    int32_t             elementSize {4}; // 4 for float32, 8 for float64
    std::vector<double> value;           // widened, the transforms are done in double anyway
};

std::map<std::string, Blob> readWts(std::string const &path)
{
    std::ifstream in(path);
    int32_t       count {0};
    in >> count;
    std::map<std::string, Blob> result;
    for (int32_t i = 0; i < count; ++i)
    {
        std::string name;
        int64_t     n {0};
        int32_t     elementSize {0};
        in >> name >> n >> elementSize;
        Blob blob;
        blob.elementSize = elementSize;
        blob.value.resize(n);
        for (int64_t j = 0; j < n; ++j)
        {
            std::string hex;
            in >> hex;
            uint64_t bits {0};
            std::istringstream(hex) >> std::hex >> bits;
            if (elementSize == 8)
            {
                double d;
                std::memcpy(&d, &bits, 8);
                blob.value[j] = d;
            }
            else
            {
                uint32_t b32 = static_cast<uint32_t>(bits);
                float    f;
                std::memcpy(&f, &b32, 4);
                blob.value[j] = f;
            }
        }
        result[name] = std::move(blob);
    }
    return result;
}

std::vector<float> readFloatText(std::string const &path)
{
    std::ifstream in(path);
    size_t        n {0};
    in >> n;
    std::vector<float> v(n);
    for (auto &x : v)
    {
        in >> x;
    }
    return v;
}

// ------------------------------------------------------------------------------------------------
// Replaying one table entry against a fresh set of ONNX initializers
// ------------------------------------------------------------------------------------------------

std::vector<float> replay(Entry const &e, std::map<std::string, Blob> const &wts)
{
    using Kind = nvonnxparser::RefitTransformKind;
    std::vector<float> out(static_cast<size_t>(e.count));

    switch (static_cast<Kind>(e.kind))
    {
    case Kind::kIDENTITY:
    case Kind::kDOUBLE_TO_FLOAT:
    {
        // Both are a straight copy once the .wts reader has widened everything to double; the
        // difference is only which ONNX type the source had.
        auto const &src = wts.at(e.sources[0]).value;
        for (size_t i = 0; i < out.size(); ++i)
        {
            out[i] = static_cast<float>(src[i]);
        }
        break;
    }
    case Kind::kBATCH_NORM_FOLD_SCALE:
    {
        // combinedScale[i] = scale[i] / sqrt(variance[i] + epsilon)
        auto const &scale = wts.at(e.sources[0]).value;
        auto const &var   = wts.at(e.sources[3]).value;
        for (size_t i = 0; i < out.size(); ++i)
        {
            out[i] = static_cast<float>(scale[i] / std::sqrt(var[i] + e.epsilon));
        }
        break;
    }
    case Kind::kBATCH_NORM_FOLD_BIAS:
    {
        // combinedBias[i] = bias[i] - mean[i] * combinedScale[i]
        auto const &scale = wts.at(e.sources[0]).value;
        auto const &bias  = wts.at(e.sources[1]).value;
        auto const &mean  = wts.at(e.sources[2]).value;
        auto const &var   = wts.at(e.sources[3]).value;
        for (size_t i = 0; i < out.size(); ++i)
        {
            double combinedScale = scale[i] / std::sqrt(var[i] + e.epsilon);
            out[i]               = static_cast<float>(bias[i] - mean[i] * combinedScale);
        }
        break;
    }
    case Kind::kCONSTANT_NODE:
    case Kind::kCONSTANT_OF_SHAPE:
    {
        // Not an initializer at all: the bytes come from the node attribute, and the observer
        // handed them to us at build time. Nothing to look up.
        assert(e.fixedData.size() >= out.size() * sizeof(float));
        std::memcpy(out.data(), e.fixedData.data(), out.size() * sizeof(float));
        break;
    }
    }
    return out;
}

// ------------------------------------------------------------------------------------------------

int runEngine(ICudaEngine &engine, std::vector<float> const &input, std::vector<float> &output)
{
    auto        context    = std::unique_ptr<IExecutionContext>(engine.createExecutionContext());
    char const *inputName  = engine.getIOTensorName(0);
    char const *outputName = engine.getIOTensorName(1);

    void *deviceInput {nullptr}, *deviceOutput {nullptr};
    CHECK(cudaMalloc(&deviceInput, input.size() * sizeof(float)));
    CHECK(cudaMalloc(&deviceOutput, output.size() * sizeof(float)));
    CHECK(cudaMemcpy(deviceInput, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    context->setTensorAddress(inputName, deviceInput);
    context->setTensorAddress(outputName, deviceOutput);
    cudaStream_t stream {nullptr};
    CHECK(cudaStreamCreate(&stream));
    context->enqueueV3(stream);
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(output.data(), deviceOutput, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(deviceInput));
    CHECK(cudaFree(deviceOutput));
    CHECK(cudaStreamDestroy(stream));
    return 0;
}

float maxAbsDiff(std::vector<float> const &a, std::vector<float> const &b)
{
    float worst = 0.0F;
    for (size_t i = 0; i < a.size(); ++i)
    {
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

int main()
{
    printf("TensorRT headers %d.%d.%d.%d, runtime %d\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD, getInferLibVersion());

    // --------------------------------------------------------------------------------------------
    printf("\n============================== Start [build machine]\n");
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(
        1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    auto parser  = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser->parseFromFile("model-v1.onnx", static_cast<int>(ILogger::Severity::kERROR)))
    {
        printf("    failed parsing model-v1.onnx\n");
        return 1;
    }
    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    config->setFlag(BuilderFlag::kREFIT);
    // TF32 is on by default and puts ~3e-04 between this engine and onnxruntime, which would
    // swamp the thing being checked here. See 02-API/Network/weight_transport.py.
    config->clearFlag(BuilderFlag::kTF32);
    auto plan = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        printf("    failed building the engine\n");
        return 1;
    }
    printf("    refittable plan: %zu bytes\n", plan->size());

    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    auto engine  = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));

    // Without the observer, this is all the deploy side would have to go on.
    {
        auto                      refitter = std::unique_ptr<IRefitter>(createInferRefitter(*engine, gLogger));
        int32_t                   nAll     = refitter->getAllWeights(0, nullptr);
        std::vector<char const *> names(nAll);
        refitter->getAllWeights(nAll, names.data());
        printf("\n    getAllWeights() reports %d refittable weights:\n      ", nAll);
        for (auto n : names)
        {
            printf("%s  ", n);
        }
        printf("\n    Three of those names appear nowhere in the ONNX file. Knowing which\n");
        printf("    initializers feed them, and how, is exactly what the observer supplies.\n");
    }

    // Attach the observer. It fires once per refittable weight, during refitFromFile.
    auto refitter       = std::unique_ptr<IRefitter>(createInferRefitter(*engine, gLogger));
    auto parserRefitter = std::unique_ptr<nvonnxparser::IParserRefitter>(
        nvonnxparser::createParserRefitter(*refitter, gLogger));
    TableRecorder recorder;
    parserRefitter->setRefitObserver(&recorder);
    if (!parserRefitter->refitFromFile("model-v1.onnx"))
    {
        printf("    refitFromFile failed\n");
        return 1;
    }
    printf("\n    the observer recorded %zu entries:\n", recorder.table.size());
    printf("    %-26s %-24s %6s  %-9s  sources\n", "engine weight", "transform", "count", "epsilon");
    printf("    %s\n", std::string(96, '-').c_str());
    for (auto const &e : recorder.table)
    {
        printf("    %-26s %-24s %6ld  ", e.trtName.c_str(), kindName(static_cast<nvonnxparser::RefitTransformKind>(e.kind)), e.count);
        if (e.epsilon != 0.0F)
        {
            printf("%-9.3g  ", e.epsilon);
        }
        else
        {
            printf("%-9s  ", "-");
        }
        if (!e.fixedData.empty())
        {
            printf("(%zu bytes carried in the record)", e.fixedData.size());
        }
        for (auto const &s : e.sources)
        {
            printf("%s ", s.c_str());
        }
        printf("\n");
    }
    writeTable(recorder.table, "refit-table.txt");
    std::ifstream tableFile("refit-table.txt", std::ios::ate | std::ios::binary);
    printf("\n    wrote refit-table.txt (%ld bytes). Together with the plan, that is the whole\n",
           static_cast<long>(tableFile.tellg()));
    printf("    deploy-time dependency -- no .onnx, no parser, no protobuf.\n");

    std::ofstream planFile("model.plan", std::ios::binary);
    planFile.write(static_cast<char const *>(plan->data()), plan->size());
    planFile.close();
    printf("============================== End   [build machine]\n");

    // --------------------------------------------------------------------------------------------
    printf("\n============================== Start [deploy machine]\n");
    auto               input       = readFloatText("input.txt");
    auto               referenceV1 = readFloatText("reference-v1.txt");
    auto               referenceV2 = readFloatText("reference-v2.txt");
    std::vector<float> output(referenceV1.size());

    if (runEngine(*engine, input, output) != 0)
    {
        return 1;
    }
    printf("    engine as built      vs reference-v1: max |diff| = %.3e\n", maxAbsDiff(output, referenceV1));

    // Everything below reads only refit-table.txt and model-v2.wts.
    auto table = readTable("refit-table.txt");
    auto wts   = readWts("model-v2.wts");
    printf("    replaying %zu entries against model-v2.wts (%zu blobs), no ONNX parser involved\n",
           table.size(),
           wts.size());

    auto                            deployRefitter = std::unique_ptr<IRefitter>(createInferRefitter(*engine, gLogger));
    std::vector<std::vector<float>> keepAlive; // Weights does not own its buffer
    for (auto const &e : table)
    {
        keepAlive.push_back(replay(e, wts));
        Weights w {static_cast<DataType>(e.trtDtype), keepAlive.back().data(), static_cast<int64_t>(keepAlive.back().size())};
        if (!deployRefitter->setNamedWeights(e.trtName.c_str(), w))
        {
            printf("    setNamedWeights failed for %s\n", e.trtName.c_str());
            return 1;
        }
    }
    int32_t nMissing = deployRefitter->getMissing(0, nullptr, nullptr);
    printf("    missing weights after replay: %d\n", nMissing);
    if (!deployRefitter->refitCudaEngine())
    {
        printf("    refitCudaEngine failed\n");
        return 1;
    }

    if (runEngine(*engine, input, output) != 0)
    {
        return 1;
    }
    float const againstV2 = maxAbsDiff(output, referenceV2);
    float const againstV1 = maxAbsDiff(output, referenceV1);
    printf("    engine after replay  vs reference-v2: max |diff| = %.3e\n", againstV2);
    printf("    engine after replay  vs reference-v1: max |diff| = %.3e   (it really moved)\n", againstV1);
    printf("============================== End   [deploy machine]\n");

    if (againstV2 > 1e-5F)
    {
        printf("\nFAILED: the replayed refit does not match onnxruntime on model-v2\n");
        return 1;
    }
    printf("\n    The BatchNormalization fold is the entry worth staring at. TensorRT has no\n");
    printf("    BatchNormalization layer, so the parser folds four ONNX initializers into two\n");
    printf("    engine weights named tmp_refittable_weight*, using an epsilon that lives in a\n");
    printf("    node attribute. Reproducing that by hand means re-deriving the parser's fusion\n");
    printf("    rules and tracking them across releases. The observer just tells you.\n");
    printf("\nFinish\n");
    return 0;
}
