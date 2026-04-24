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

#include "cnpy.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace nvinfer1;

class CookbookCalibratorV1 : public IInt8EntropyCalibrator2
{
private:
    int                nCalibration {0};
    int                nElement {0};
    size_t             bufferSize {0};
    int                nBatch {0};
    int                iBatch {0};
    std::vector<float> hostData;
    float             *bufferD {nullptr};
    Dims64             dim;
    std::string        cacheFile {""};
    std::vector<char>  cacheData;

public:
    CookbookCalibratorV1(std::string const &calibrationDataFile, int const nCalibration, Dims64 const inputShape, std::string const &cacheFile);
    ~CookbookCalibratorV1() noexcept;
    int32_t     getBatchSize() const noexcept;
    bool        getBatch(void *bindings[], char const *names[], int32_t nbBindings) noexcept;
    void const *readCalibrationCache(std::size_t &length) noexcept;
    void        writeCalibrationCache(void const *ptr, std::size_t length) noexcept;
};
