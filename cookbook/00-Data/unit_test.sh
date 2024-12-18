#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
set -x
#clear

python3 extract_MNIST.py
python3 get_model_part1.py
python3 get_model_part2.py
rm -rf *.pkl

# Do not remove files after unit tests
#if [ $TRT_COOKBOOK_CLEAN ]; then
#    rm -rf data/test data/train data/*.npy data/*.npz models/*.onnx models/*.weight model/*npz models/*.pth
#fi

echo "Finish `basename $(pwd)`"
