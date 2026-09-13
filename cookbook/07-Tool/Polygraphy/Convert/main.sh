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

rm -rf *.Int8Cache *.json *.lock *.log *.onnx *.TimingCache *.trt

export MODEL_TRAINED=${TRT_COOKBOOK_PATH}/00-Data/model/model-trained.onnx
export MODEL_HALF_MNIST=${TRT_COOKBOOK_PATH}/00-Data/model/model-half-mnist.onnx

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
    --trt-min-shapes 'x:[1,1,28,28]' \
    --trt-opt-shapes 'x:[4,1,28,28]' \
    --trt-max-shapes 'x:[16,1,28,28]' \
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

# 04-Convert a ONNX file into another ONNX file
# Notice:
# + `--convert-to onnx` is the ONNX -> ONNX direction. It only loads and saves the model, plus the
#   options of the ONNX loader / saver: `--shape-inference` here, `--fp-to-fp16` below,
#   `--save-external-data` to split the weights out.
# + There is NO constant folding here (`--fold-constants` is rejected as an unrecognized option),
#   graph surgery lives in `polygraphy surgeon sanitize`, see ../Surgeon/main.sh.
polygraphy convert \
    $MODEL_TRAINED \
    --convert-to onnx \
    --shape-inference \
    --output model-trained-shape-inferred.onnx \
    > result-04.log 2>&1

# 05-Cast a ONNX file to float16
# + `--fp-to-fp16` casts every float32 initializer / tensor to float16 *in the ONNX file*. On
#   TensorRT 11 this is how a half-precision model is expressed, since `--fp16` (a builder flag)
#   was removed, see ../More/05-BuildNetworkByHand/ and ../More/13-PerLayerPrecision/.
polygraphy convert \
    $MODEL_TRAINED \
    --convert-to onnx \
    --fp-to-fp16 \
    --output model-trained-fp16.onnx \
    > result-05.log 2>&1

ls -l $MODEL_TRAINED model-trained-shape-inferred.onnx model-trained-fp16.onnx >> result-05.log 2>&1

# What actually changed: the 8 float initializers became float16 (the 9th is the int64 shape),
# and two `Cast` nodes appeared, because the graph *inputs and outputs stay float32*.
# So the engine's I/O dtypes are unchanged and only the arithmetic inside is half precision.
polygraphy inspect model model-trained-fp16.onnx --show layers attrs >> result-05.log 2>&1

# Building it needs no builder flag at all: the types come from the file, which is the whole point
# of a strongly typed network (see ../More/05-BuildNetworkByHand/).
polygraphy run \
    model-trained-fp16.onnx \
    --trt \
    --input-shapes 'x:[1,1,28,28]' \
    >> result-05.log 2>&1

echo "Finish"
