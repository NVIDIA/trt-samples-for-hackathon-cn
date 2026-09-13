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

rm -rf replays/ polygraphy_artifacts/ *.engine *.json *.log *.onnx flaky-artifact.txt iteration-counter.txt

export MODEL_TRAINED=${TRT_COOKBOOK_PATH}/00-Data/model/model-trained.onnx
export MODEL_UNKNOWN=${TRT_COOKBOOK_PATH}/00-Data/model/model-unknown.onnx

# 01-Find the first failed subgraph
polygraphy debug reduce \
    $MODEL_UNKNOWN \
    --output reduced.onnx \
    --model-input-shapes 'inputT0:[1,1,28,28]' \
    --check polygraphy run --trt \
    > result-01.log 2>&1

# 02-Reduce Failing ONNX Models
# + Just shows the process of search failed node, but the model used in this example is no problem
polygraphy run \
    $MODEL_TRAINED \
    --onnxrt \
    --onnx-outputs mark all \
    --save-inputs model-trained-input.json \
    --save-outputs model-trained-output.json \
    > result-02.log 2>&1

polygraphy data merge \
    model-trained-input.json \
    model-trained-output.json \
    -o model-trained-io.json
    >> result-02.log 2>&1

polygraphy debug reduce \
    $MODEL_TRAINED \
    --mode bisect \
    -o model-trained-reduce.onnx \
    --load-inputs model-trained-io.json \
    --check polygraphy run polygraphy_debug.onnx --trt --load-inputs model-trained-io.json --load-outputs model-trained-output.json
    >> result-02.log 2>&1

# 03-Reduce Failing ONNX Models
# + Continue of the example above, and pretend we have a bad node of type "Gemm"
polygraphy debug reduce \
    $MODEL_TRAINED \
    --mode bisect \
    --fail-regex "Op: Gemm" \
    -o model-trained-reduce-gemm.onnx \
    --check polygraphy inspect model polygraphy_debug.onnx --show layers \
    > result-03.log 2>&1

# 04-Build the same engine several times and sort the artifacts by the result of a check
# + This is the tool for *flaky* builds: TensorRT picks tactics by timing, so two builds of the same
#   network are not guaranteed to be the same engine. `debug build` repeats the build and runs `--check`
#   on each resulting engine, moving the artifacts named by `--artifacts` into `good` or `bad`.
# + `--until` is mandatory: an int (fixed number of iterations), or `good` / `bad` to stop at the first one.
# + The sorted artifacts land in `polygraphy_artifacts/{good,bad}/` (`--art-dir` to change it), each one
#   stamped with its iteration number. NOT in `good/` and `bad/` of the current directory.
# + Historically this was paired with `--save-tactics` to name the guilty tactic. That path is gone on
#   TensorRT 11 (`--save-tactics` raises `AttributeError: ... has no attribute 'IAlgorithmSelector'`),
#   see ../More/12-TacticsAndReproducibility/ for the timing-cache based replacement.
polygraphy debug build \
    $MODEL_TRAINED \
    --until 3 \
    --artifacts polygraphy_debug.engine \
    --check polygraphy run polygraphy_debug.engine --model-type engine --trt --input-shapes 'x:[1,1,28,28]' \
    > result-04.log 2>&1

ls -lR polygraphy_artifacts/ >> result-04.log 2>&1 || true

# 05-Repeat an arbitrary command and sort its artifacts the same way
# + `debug repeat` knows nothing about models: it runs whatever `--check` says, N times.
#   Anything that is flaky and produces a file can be sorted with it.
# + The check below fails on every third iteration. A counter file stands in for the real flakiness
#   so that this reference log stays reproducible; the artifacts of the failing iterations end up in
#   `polygraphy_artifacts/bad/` and the others in `polygraphy_artifacts/good/`.
echo 0 > iteration-counter.txt

polygraphy debug repeat \
    --until 6 \
    --artifacts flaky-artifact.txt \
    --check bash -c 'n=$(($(cat iteration-counter.txt) + 1)); echo $n > iteration-counter.txt; echo "iteration $n" > flaky-artifact.txt; test $((n % 3)) -ne 0' \
    > result-05.log 2>&1 || true

ls -lR polygraphy_artifacts/ >> result-05.log 2>&1 || true

echo "Finish"
