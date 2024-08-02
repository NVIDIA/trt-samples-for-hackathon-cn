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

#include <string>
#include <vector>

using namespace cnpy;

int main()
{
    std::string inputNpyFile = "data.npy";
    std::string inputNpzFile = "data.npz";
    std::string inputVarName = "aa";

    NpyArray npy = npy_load(inputNpyFile);
    printf("%s:\n", inputNpyFile.c_str());
    for (int i = 0; i < npy.num_vals; ++i)
    {
        printf("%3.0f,", npy.data<float>()[i]);
    }
    printf("\n");

    npz_t npz = npz_load(inputNpzFile);
    npy       = npz[inputVarName];
    printf("%s[%s]:\n", inputNpyFile.c_str(), inputVarName.c_str());
    for (int i = 0; i < npy.num_vals; ++i)
    {
        printf("%3.0f,", npy.data<float>()[i]);
    }
    printf("\n");

    std::vector<int>    bb(6, 0);
    std::vector<size_t> shape {2, 3};
    std::string         outputNpyFile = "output-data.npy";
    std::string         outputNpzFile = "output-data.npz";
    std::string         outputVarName = "b";

    npy_save(outputNpyFile, bb.data(), shape);
    npz_save(outputNpzFile, outputVarName, bb.data(), shape);

    printf("Finish\n");
    return 0;
}
