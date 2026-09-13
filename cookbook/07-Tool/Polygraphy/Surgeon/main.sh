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

rm -rf *.log *.onnx

# 00-Simplify the graph using polygraphy (the most common usegae)
# If we provide more information (such as static batch-size), we can see the ONNX is significantly simplified.
export MODEL_TRAINED=${TRT_COOKBOOK_PATH}/00-Data/model/model-trained.onnx
export MODEL_REDUNDANT=${TRT_COOKBOOK_PATH}/00-Data/model/model-redundant.onnx

polygraphy surgeon sanitize $MODEL_REDUNDANT \
    --cleanup \
    --fold-constant \
    --toposort \
    -o model-redundant-FC-DynamicBatch.onnx \
    > result-00.log

polygraphy surgeon sanitize $MODEL_REDUNDANT \
    --cleanup \
    --fold-constant \
    --toposort \
    --override-input-shapes inputT0:[7,2,3,4] \
    -o model-redundant-FC-StaticBatch.onnx \
    > result-01.log

# 02-Extract a subgraph from ONNX
polygraphy surgeon extract $MODEL_REDUNDANT \
    --inputs "inputT0:[nBS,2,3,4]:float32" \
    --outputs "RedundantModel-V-6-Concat-0:auto" \
    -o model-redundant-EX.onnx \
    > result-02.log

# 03-Insert a node into ONNX
polygraphy surgeon insert $MODEL_REDUNDANT \
    --name "MyNewNode" \
    --op "NewNode" \
    --inputs "RedundantModel-V-1-ReduceProd-0" \
    --outputs "RedundantModel-V-1-ReduceProd-0" \
    --attrs arg_int=31193 arg_float=3.14 arg_str=wili arg_list=[0,1,2] \
    -o model-redundant-IN.onnx \
    > result-03.log

# 04-Prune a ONNX to support sparisty in TensorRT
# In this example, our model is pruned successfully but not adpoted in engine finally due to performance.
polygraphy surgeon prune $MODEL_TRAINED \
    -o model-trained-PR.onnx \
    > result-04.log

cat result-04.log | grep pruning

polygraphy run model-trained-PR.onnx \
    --trt \
    --sparse-weights \
    --verbose \
    | grep Sparsity && true

# 05-Strip the weights out of a ONNX file
# + This is the ONNX side of the weight-stripped engine workflow: the graph, shapes and attributes are kept,
#   the initializer payloads are dropped, so the file can be shipped / diffed without the weights.
# + `--exclude-list` takes a text file of initializer names which must keep their data.
polygraphy surgeon weight-strip $MODEL_TRAINED \
    -o model-trained-WS.onnx \
    > result-05.log 2>&1

ls -l $MODEL_TRAINED model-trained-WS.onnx >> result-05.log 2>&1

# 06-Reconstruct proxy weights of a stripped ONNX file
# Notice:
# + The reconstructed weights are *proxies* (they make the file loadable and buildable again),
#   NOT the original values, so the outputs of the reconstructed model are meaningless.
#   Its use is building an engine / measuring performance / inspecting the network without shipping weights.
polygraphy surgeon weight-reconstruct model-trained-WS.onnx \
    -o model-trained-WR.onnx \
    > result-06.log 2>&1

ls -l model-trained-WS.onnx model-trained-WR.onnx >> result-06.log 2>&1

# The stripped model has no initializer data left, the reconstructed one does
polygraphy inspect model model-trained-WS.onnx --show weights >> result-06.log 2>&1
polygraphy inspect model model-trained-WR.onnx --show weights >> result-06.log 2>&1

echo "Finish"
