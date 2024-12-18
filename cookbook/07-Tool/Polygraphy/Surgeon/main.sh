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
rm -rf *.log *.onnx
#clear

# 00-Simplify the graph using polygraphy (the most common usegae)
# If we provide more information (such as static batch-size), we can see the ONNX is significantly simplified.
export MODEL_TRAINED=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx
export MODEL_REDUNDANT=$TRT_COOKBOOK_PATH/00-Data/model/model-redundant.onnx

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
    | grep Sparsity

echo "Finish"
