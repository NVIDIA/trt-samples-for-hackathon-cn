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
#

set -xeuo pipefail

rm -rf *.log *.onnx *.trt modify_config.py modify_network.py modify_onnx.py

export MODEL_TRAINED=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx

# 01-Create a script to modify the network
polygraphy template trt-network \
    $MODEL_TRAINED \
    --output modify_network.py

# Once we finish the edition, we can use convert mode to build the TensorRT engine
polygraphy convert \
    modify_network.py \
    --convert-to trt \
    --output "./model-trained.trt" \
    --model-type=trt-network-script \
    > result-01.log 2>&1

#02-Create a script to modify the config in TensorRT. TODO: how to use it?
polygraphy template trt-config \
    $MODEL_TRAINED \
    --output modify_config.py

#03-Create a script to modify ONNX file
polygraphy template onnx-gs \
    $MODEL_TRAINED \
    --output modify_onnx.py

echo "Finish"
