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

rm -rf *.json *.lock *.log *.onnx *.TimingCache *.trt

export MODEL_TRAINED=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx
export MODEL_HALF_MNIST=$TRT_COOKBOOK_PATH/00-Data/model/model-half-mnist.onnx

# 01-Parse ONNX file, build and save TensorRT engine without any more option
polygraphy convert \
    $MODEL_TRAINED \
    --convert-to trt \
    --output ./model-trained-0.trt \
    > result-01.log 2>&1

# 02-Parse ONNX file, build and save TensorRT engine with more regular options (see Help.txt to get more parameters)
# + For the shape option, use "," to separate dimensions and use " " to separate the tensors (which is different from `trtexec`)
# + e.g. "--trt-min-shapes 'x:[16,320,256]' 'y:[8,4]' 'z:[]'"
# + Timing cache can be reused with `--load-timing-cache` during rebuild
# + More than one combination of `--trt-*-shapes` can be used for multiple optimization-profile
polygraphy convert \
    $MODEL_TRAINED \
    --convert-to trt \
    --output ./model-trained.trt \
    --save-timing-cache model-trained.TimingCache \
    --save-tactics model-trained-tactics.json \
    --trt-min-shapes 'x:[1,1,28,28]' \
    --trt-opt-shapes 'x:[4,1,28,28]' \
    --trt-max-shapes 'x:[16,1,28,28]' \
    --fp16 \
    --memory-pool-limit workspace:1G \
    --builder-optimization-level 3 \
    --max-aux-streams 4 \
    --verbose \
    > result-02.log 2>&1

# 03-Convert a TensorRT network into a ONNX-like file for visualization in Netron
# Here is a error to convert model-trained.onnx:
# + ValueError: Could not infer attribute `reshape_dims` type from empty iterator), so we use another model
polygraphy convert \
    $MODEL_HALF_MNIST \
    --convert-to onnx-like-trt-network \
    --output model-half-mnist-network.onnx \
    > result-03.log 2>&1

# 02-Parse ONNX file, build and save TensorRT engine in INT8 mode
# + We need to provide a script to load calibration data
# + INT8 cache can be reused with `---calibration-cache` during rebuild
polygraphy convert \
    $MODEL_TRAINED \
    --convert-to trt \
    --output model-trained-int8.trt \
    --int8 \
    --data-loader-script data_loader.py \
    --calibration-cache model-trained.Int8Cache \
    > result-03.log 2>&1

echo "Finish"
