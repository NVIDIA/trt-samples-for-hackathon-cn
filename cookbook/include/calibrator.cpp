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

#include "calibrator.h"

using namespace nvinfer1;

MyCalibratorV1::MyCalibratorV1(const std::string &calibrationDataFile, const int nCalibration, const Dims64 dim, const std::string &cacheFile):
    nCalibration(nCalibration), dim(dim), cacheFile(cacheFile), iBatch(0)
{
    //cnpy::npz_t    npzFile = cnpy::npz_load(calibrationDataFile);
    //cnpy::NpyArray array   = npzFile[std::string("calibrationData")];
    cnpy::NpyArray array = cnpy::npy_load(calibrationDataFile);
    pData                = array.data<float>();
    if (pData == nullptr)
    {
        std::cout << "Fail getting calibration data" << std::endl;
        return;
    }

    nElement = 1;
    for (int i = 0; i < dim.nbDims; ++i)
    {
        nElement *= dim.d[i];
    }
    bufferSize = sizeof(float) * nElement;
    nBatch     = array.num_bytes() / bufferSize;

    cudaMalloc((void **)&bufferD, bufferSize);

    return;
}

MyCalibratorV1::~MyCalibratorV1() noexcept
{
    if (bufferD != nullptr)
    {
        cudaFree(bufferD);
    }
    return;
}

int32_t MyCalibratorV1::getBatchSize() const noexcept
{
    return dim.d[0];
}

bool MyCalibratorV1::getBatch(void *bindings[], char const *names[], int32_t nbBindings) noexcept
{
    if (iBatch < nBatch)
    {
        cudaMemcpy(bufferD, &pData[iBatch * nElement], bufferSize, cudaMemcpyHostToDevice);
        bindings[0] = bufferD;
        iBatch++;
        return true;
    }
    else
    {
        return false;
    }
}

void const *MyCalibratorV1::readCalibrationCache(std::size_t &length) noexcept
{
    std::fstream f;
    f.open(cacheFile, std::fstream::in);
    if (f.fail())
    {
        std::cout << "Fail finding cache file" << std::endl;
        return nullptr;
    }
    char *ptr = new char[length];
    if (f.is_open())
    {
        f >> ptr;
    }
    return ptr;
}

void MyCalibratorV1::writeCalibrationCache(void const *ptr, std::size_t length) noexcept
{
    std::ofstream f(cacheFile, std::ios::binary);
    if (f.fail())
    {
        std::cout << "Fail opening cache file to write" << std::endl;
        return;
    }
    f.write(static_cast<char const *>(ptr), length);
    if (f.fail())
    {
        std::cout << "Fail saving cache file" << std::endl;
        return;
    }
    f.close();
}
