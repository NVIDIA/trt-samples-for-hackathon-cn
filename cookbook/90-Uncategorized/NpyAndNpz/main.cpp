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

#include <filesystem>

using namespace cnpy;

template<typename T>
void printNpy(const NpyArray &npy, std::string name = std::string(""))
{
    printf("NpyArray %s:\n", name.c_str());
    printf("fortran_order   :%b\n", npy.fortran_order);
    printf("num_vals        :%ld elements\n", npy.num_vals);
    printf("word_size       :%ld Byte/element\n", npy.word_size);
    printf("shape           :[");
    for (size_t const &i : npy.shape)
    {
        printf("%lu, ", i);
    }
    printf("]\n");
    printf("data            :\n");
    for (int i = 0; i < npy.num_vals; ++i)
    {
        if constexpr (std::is_same_v<T, int>)
        {
            printf("%4d,", npy.data<int>()[i]);
        }
        else
        {
            printf("%4.1f,", npy.data<float>()[i]);
        }
    }
    printf("\n\n");
}

int main()
{
    // Load data from npy and npz files
    std::string const inputNpyFile = "input_data.npy";
    std::string const inputNpzFile = "input_data.npz";
    std::string const inputKey0    = "data0";
    std::string const inputKey1    = "data1";

    // Find if the input files exist
    if (!(std::filesystem::exists(inputNpyFile) && std::filesystem::exists(inputNpzFile)))
    {
        printf("Input files do not exist, run `python get_data.py` at first.\n");
        return 1;
    }

    NpyArray npy = npy_load(inputNpyFile);
    printNpy<float>(npy);

    npz_t npz = npz_load(inputNpzFile);
    printNpy<float>(npz[inputKey0], inputKey0);
    printNpy<int>(npz[inputKey1], inputKey1);

    // Save data to npy and npz files
    std::vector<float> data0(60);
    std::iota(data0.begin(), data0.end(), 0);
    std::vector<size_t> data0_shape {3, 4, 5};
    std::vector<int>    data1(4, 0);
    std::vector<size_t> data1_shape {2, 2};

    std::string const outputNpyFile = "output_data.npy";
    std::string const outputNpzFile = "output_data.npz";
    std::string const outputKey0    = "data0";
    std::string const outputKey1    = "data1";

    npy_save(outputNpyFile, data0.data(), data0_shape, "w");
    npz_save(outputNpzFile, outputKey0, data0.data(), data0_shape, "w");
    npz_save(outputNpzFile, outputKey1, data1.data(), data1_shape, "a");

    printf("Finish\n");
    return 0;
}
