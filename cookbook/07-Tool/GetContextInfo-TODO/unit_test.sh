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

# Build a engine with 2 optimization profiles (just for this example, 1 is enough in normal use case).
trtexec \
    --onnx=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx \
    --saveEngine=model.trt \
    --profile=0 \
        --minShapes=x:1x1x28x28 \
        --optShapes=x:4x1x28x28 \
        --maxShapes=x:16x1x28x28 \
    --profile=1 \
        --minShapes=x:8x1x28x28 \
        --optShapes=x:32x1x28x28 \
        --maxShapes=x:64x1x28x28 \
    --fp16 \
    --noTF32 \
    --memPoolSize=workspace:1024MiB \
    --builderOptimizationLevel=0 \
    --skipInference \
    --verbose

python3 main.py -i model.trt > log-main.py.log

make test

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.trt *.log
fi

echo "Finish `basename $(pwd)`"
