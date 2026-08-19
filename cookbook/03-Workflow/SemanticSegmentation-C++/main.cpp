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

//! A semantic-segmentation runtime in C++: image in, class map out.
//!
//! The cookbook's other C++ examples (`01-SimpleDemo`, `03-Workflow/*/C++`) feed a tensor in
//! and print a tensor out. A deployed vision service does neither: it reads an image, and the
//! interesting work is on both sides of `enqueueV3`.
//!
//! + **Pre-processing**: interleaved 8-bit RGB has to become planar float NCHW, scaled to
//!   [0, 1] and normalised per channel. That layout change is the single most common source of
//!   "the engine runs but the answers are nonsense".
//! + **Post-processing**: the engine emits `[1, N_CLASS, H, W]` logits, not a picture. Turning
//!   that into a class map is a per-pixel argmax *across the channel stride*, and then into an
//!   image via a colour palette.
//!
//! Re-expressed from `quickstart/SemanticSegmentation/tutorial-runtime.cpp` in TensorRT-OSS,
//! which does the same thing for FCN-ResNet101. The upstream file is already written against
//! the modern API (`setTensorAddress` + `enqueueV3`); what it needs is `samples/common`, a
//! 200 MB torchvision download and its own PPM helpers, so this version is standalone and
//! checks its answer against a NumPy reference instead of asking you to look at a picture.
//!
//! Run `python3 export.py` first to produce the ONNX file and the input image.

#include "cookbookHelper.cuh"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <sstream>

using namespace nvinfer1;

static Logger gLogger(ILogger::Severity::kERROR);

std::string const kOnnxFile {"model-segmentation.onnx"};
std::string const kInputImage {"data-input.ppm"};
std::string const kOutputImage {"data-output.ppm"};
std::string const kOutputRaw {"data-output.raw"};

int32_t const kClass {4};
// The same normalisation `export.py` applied, and the reason it has to be repeated here: the
// constants live in the training script, never in the ONNX file, so they are the classic thing
// to get silently wrong when the two halves are written by different people.
float const kMean[3] {0.485F, 0.456F, 0.406F};
float const kStd[3] {0.229F, 0.224F, 0.225F};

//! The palette used to turn a class index into something a human can look at.
uint8_t const kPalette[4][3] {{220, 50, 50}, {50, 220, 50}, {50, 50, 220}, {220, 220, 50}};

//! Read a binary PPM (P6) into interleaved RGB.
bool readPPM(std::string const &path, std::vector<uint8_t> &pixel, int32_t &width, int32_t &height)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::cout << "Failed opening " << path << ", run `python3 export.py` first" << std::endl;
        return false;
    }
    std::string magic;
    int32_t     maxValue {0};
    file >> magic >> width >> height >> maxValue;
    if (magic != "P6" || maxValue != 255)
    {
        std::cout << "Only binary PPM (P6) with maxValue 255 is supported, got " << magic << std::endl;
        return false;
    }
    file.get(); // The single whitespace byte between the header and the data
    pixel.resize(static_cast<size_t>(width) * height * 3);
    file.read(reinterpret_cast<char *>(pixel.data()), pixel.size());
    return file.gcount() == static_cast<std::streamsize>(pixel.size());
}

bool writePPM(std::string const &path, std::vector<uint8_t> const &pixel, int32_t width, int32_t height)
{
    std::ofstream file(path, std::ios::binary);
    if (!file)
    {
        return false;
    }
    file << "P6\n"
         << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<char const *>(pixel.data()), pixel.size());
    return true;
}

int main()
{
    // ---- Read the image
    std::vector<uint8_t> imagePixel;
    int32_t              width {0}, height {0};
    if (!readPPM(kInputImage, imagePixel, width, height))
    {
        return 1;
    }
    std::cout << "Input image: " << width << "x" << height << ", " << imagePixel.size() << " bytes interleaved RGB" << std::endl;

    // ---- Build the engine from ONNX
    IBuilder              *builder = createInferBuilder(gLogger);
    INetworkDefinition    *network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
    IBuilderConfig        *config  = builder->createBuilderConfig();
    nvonnxparser::IParser *parser  = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile(kOnnxFile.c_str(), static_cast<int32_t>(ILogger::Severity::kERROR)))
    {
        std::cout << "Failed parsing " << kOnnxFile << std::endl;
        return 1;
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
    std::cout << "Engine built, " << engine->getNbIOTensors() << " IO tensors" << std::endl;

    char const *inputName {nullptr};
    char const *outputName {nullptr};
    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i)
    {
        char const *name                                                                 = engine->getIOTensorName(i);
        (engine->getTensorIOMode(name) == TensorIOMode::kINPUT ? inputName : outputName) = name;
    }
    Dims64 const inputShape  = context->getTensorShape(inputName);
    Dims64 const outputShape = context->getTensorShape(outputName);
    std::cout << "  input  " << inputName << " " << shapeToString(inputShape) << std::endl;
    std::cout << "  output " << outputName << " " << shapeToString(outputShape) << std::endl;

    int32_t const netHeight = inputShape.d[2];
    int32_t const netWidth  = inputShape.d[3];
    if (netHeight != height || netWidth != width)
    {
        std::cout << "This example assumes the image already matches the network resolution" << std::endl;
        return 1;
    }

    // ---- Pre-process: interleaved uint8 HWC -> planar float CHW, scaled and normalised
    //
    // The loop order is the whole point. The source advances by 3 bytes per pixel with the
    // channel innermost; the destination advances by 1 float per pixel with the channel
    // *outermost*, one full plane apart. Copying straight across "works" and produces a
    // picture-shaped tensor whose channels are interleaved garbage.
    size_t const       pixelCount = static_cast<size_t>(height) * width;
    std::vector<float> inputHost(pixelCount * 3);
    for (int32_t c = 0; c < 3; ++c)
    {
        for (size_t i = 0; i < pixelCount; ++i)
        {
            float const value             = static_cast<float>(imagePixel[i * 3 + c]) / 255.0F;
            inputHost[c * pixelCount + i] = (value - kMean[c]) / kStd[c];
        }
    }

    // ---- Run
    size_t const outputCount = static_cast<size_t>(kClass) * pixelCount;
    void        *inputDevice {nullptr};
    void        *outputDevice {nullptr};
    CHECK(cudaMalloc(&inputDevice, inputHost.size() * sizeof(float)));
    CHECK(cudaMalloc(&outputDevice, outputCount * sizeof(float)));
    CHECK(cudaMemcpy(inputDevice, inputHost.data(), inputHost.size() * sizeof(float), cudaMemcpyHostToDevice));
    context->setTensorAddress(inputName, inputDevice);
    context->setTensorAddress(outputName, outputDevice);

    cudaStream_t stream {nullptr};
    CHECK(cudaStreamCreate(&stream));
    if (!context->enqueueV3(stream))
    {
        std::cout << "enqueueV3 failed" << std::endl;
        return 1;
    }
    CHECK(cudaStreamSynchronize(stream));

    std::vector<float> outputHost(outputCount);
    CHECK(cudaMemcpy(outputHost.data(), outputDevice, outputCount * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- Post-process: per-pixel argmax over the channel dimension
    //
    // The logits are `[1, C, H, W]`, so the C values belonging to one pixel are `pixelCount`
    // floats apart, not adjacent. Reading them as if they were adjacent is the mirror image of
    // the pre-processing mistake, and just as silent.
    std::vector<uint8_t> classMap(pixelCount);
    for (size_t i = 0; i < pixelCount; ++i)
    {
        int32_t best {0};
        float   bestValue {outputHost[i]};
        for (int32_t c = 1; c < kClass; ++c)
        {
            float const value = outputHost[static_cast<size_t>(c) * pixelCount + i];
            if (value > bestValue)
            {
                bestValue = value;
                best      = c;
            }
        }
        classMap[i] = static_cast<uint8_t>(best);
    }

    std::vector<int32_t> histogram(kClass, 0);
    for (uint8_t value : classMap)
    {
        ++histogram[value];
    }
    std::cout << "Pixels per class:";
    for (int32_t c = 0; c < kClass; ++c)
    {
        std::cout << " [" << c << "]=" << histogram[c];
    }
    std::cout << std::endl;

    // ---- Write the class map, both as an image and as raw bytes the test can compare
    std::vector<uint8_t> colourised(pixelCount * 3);
    for (size_t i = 0; i < pixelCount; ++i)
    {
        for (int32_t c = 0; c < 3; ++c)
        {
            colourised[i * 3 + c] = kPalette[classMap[i]][c];
        }
    }
    writePPM(kOutputImage, colourised, width, height);
    std::ofstream(kOutputRaw, std::ios::binary).write(reinterpret_cast<char const *>(classMap.data()), classMap.size());
    std::cout << "Wrote " << kOutputImage << " and " << kOutputRaw << std::endl;

    // ---- Release
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(inputDevice));
    CHECK(cudaFree(outputDevice));
    delete context;
    delete engine;
    delete runtime;
    delete engineString;
    delete parser;
    delete config;
    delete network;
    delete builder;

    std::cout << "\nFinish" << std::endl;
    return 0;
}
