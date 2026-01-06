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

set -xeuo pipefail

rm -rf *.log *.onnx *.raw

export MODEL_TRAINED=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx

# 01-Save input / output data
polygraphy run \
    $MODEL_TRAINED \
    --onnxrt \
    --save-inputs model-trained-inputs.raw \
    --save-outputs model-trained-outputs.raw \
    > result-01.log 2>&1

# 02-Combine input and output data into a raw file
polygraphy data to-input \
    model-trained-inputs.raw model-trained-outputs.raw \
    --output model-trained-io.raw \
    > result-02.log 2>&1
