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

rm -rf replays/ *.json *.log *.onnx

export MODEL_TRAINED=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx
export MODEL_UNKNOWN=$TRT_COOKBOOK_PATH/00-Data/model/model-unknown.onnx

# 01-Find the first failed subgraph
polygraphy debug reduce \
    $MODEL_UNKNOWN \
    --output reduced.onnx \
    --model-input-shapes 'x:[1,1,28,28]' \
    --check polygraphy run --trt \
    > result-01.log 2>&1

# 02-Find flaky tactic
# + This example no longer works reliably
# + https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/debug/01_debugging_flaky_trt_tactics#debugging-flaky-tensorrt-tactics)
polygraphy run \
    $MODEL_TRAINED \
    --onnxrt \
    --save-outputs model-trained.json \
    > result-02.log 2>&1

polygraphy debug build \
    $MODEL_TRAINED \
    --fp16 \
    --artifacts-dir replays \
    --save-tactics replay.json \
    --artifacts replay.json \
    --until=10 \
    --check polygraphy run polygraphy_debug.engine --trt --load-outputs model-trained.json
    >> result-02.log 2>&1

# 03-Reduce Failing ONNX Models
# + Just shows the process of search failed node, but the model used in this example is no problem
polygraphy run \
    $MODEL_TRAINED \
    --onnxrt \
    --onnx-outputs mark all \
    --save-inputs model-trained-input.json \
    --save-outputs model-trained-output.json \
    > result-03.log 2>&1

polygraphy data to-input \
    model-trained-input.json \
    model-trained-output.json \
    -o model-trained-io.json
    >> result-03.log 2>&1

polygraphy debug reduce \
    $MODEL_TRAINED \
    --mode bisect \
    -o model-trained-reduce.onnx \
    --load-inputs model-trained-io.json \
    --check polygraphy run polygraphy_debug.onnx --trt --load-inputs model-trained-io.json --load-outputs model-trained-output.json
    >> result-03.log 2>&1

# 04-Reduce Failing ONNX Models
# + Continue of the example above, and pretend we have a bad node of type "Gemm"
polygraphy debug reduce \
    $MODEL_TRAINED \
    --mode bisect \
    --fail-regex "Op: Gemm" \
    -o model-trained-reduce-gemm.onnx \
    --check polygraphy inspect model polygraphy_debug.onnx --show layers \
    > result-04.log 2>&1

echo "Finish"
