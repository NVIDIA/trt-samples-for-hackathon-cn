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

int main()
{
    CHECK(cudaSetDevice(0));
    Logger                gLogger(ILogger::Severity::kERROR);
    IBuilder             *builder = createInferBuilder(gLogger);
    INetworkDefinition   *network = builder->createNetworkV2(0);
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    IBuilderConfig       *config  = builder->createBuilderConfig();

    for (auto *tensor : buildMnistNetwork(config, network, profile))
    {
        network->markOutput(*tensor);
    }

    printNetwork(network);

    return 0;
}
