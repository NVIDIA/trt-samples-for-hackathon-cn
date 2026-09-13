#!/bin/bash
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -xeuo pipefail

rm -rf *.log *.onnx *.raw

export MODEL_TRAINED=${TRT_COOKBOOK_PATH}/00-Data/model/model-trained.onnx

# 01-Save input / output data
# + `--iterations` decides how many iterations are stored in each file, which is what `data concat` glues together later
polygraphy run \
    $MODEL_TRAINED \
    --onnxrt \
    --iterations 2 \
    --save-inputs model-trained-inputs.raw \
    --save-outputs model-trained-outputs.raw \
    > result-01.log 2>&1

# 02-Combine input and output data into a raw file
# TRT-11.0 / Polygraphy 0.50: `data to-input` renamed to `data merge` (same behavior, usable with --load-inputs)
# + `merge` works across the input/output *axis*: it takes one file of inputs and one file of outputs
polygraphy data merge \
    model-trained-inputs.raw model-trained-outputs.raw \
    --output model-trained-io.raw \
    > result-02.log 2>&1

# 03-Concatenate iterations of several files of the same kind into one file
# + `concat` works across the *iteration* axis, which is the other direction than `merge`:
#   all input files must be of the same kind (all inputs, or all outputs), and the iterations are appended.
# + Typical use: several processes / machines each ran part of a dataset, and the results have to be compared at once.
polygraphy run \
    $MODEL_TRAINED \
    --onnxrt \
    --iterations 3 \
    --save-outputs model-trained-outputs-2.raw \
    >> result-01.log 2>&1

polygraphy data concat \
    model-trained-outputs.raw model-trained-outputs-2.raw \
    --output model-trained-outputs-all.raw \
    > result-03.log 2>&1

# The concatenated file holds 2 + 3 = 5 iterations
polygraphy inspect data \
    model-trained-outputs-all.raw \
    >> result-03.log 2>&1

echo "Finish"
