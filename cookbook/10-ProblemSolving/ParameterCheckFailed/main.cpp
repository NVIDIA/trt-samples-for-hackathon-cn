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

#include <NvInfer.h>
#include <iostream>

using namespace nvinfer1;

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

int main()
{
    Logger              gLogger(ILogger::Severity::kERROR);
    IBuilder *          builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    IBuilderConfig *    config  = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 30);

    ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims4 {1, 3, 600, 800});

    DimsHW  window {5, 5};
    Weights kernel {};
    Weights bias {};

    if (true) // count > 0 but values == nullptr
    {
        kernel.count  = 32 * 3 * 5 * 5;
        kernel.values = nullptr;
    }
    else // count == 0 but values != nullptr
    {
        kernel.count  = 0;
        kernel.values = (void const *)&kernel.count;
    }

    IConvolutionLayer *convLayer = network->addConvolutionNd(*inputTensor, 32, window, kernel, bias);

    network->markOutput(*convLayer->getOutput(0));
    IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
    return 0;
}
