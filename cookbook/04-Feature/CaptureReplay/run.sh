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

export TRT_SHIM_OUTPUT_JSON_FILE=$(pwd)/model-mnist.json

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtensorrt_shim.so CASE=mnist python main.py

tensorrt_player -j model-mnist.json -o model-mnist-rebuild.trt

export TRT_SHIM_OUTPUT_JSON_FILE=$(pwd)/model-pluginv2.json

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtensorrt_shim.so CASE=pluginv2 python main.py

tensorrt_player -j model-pluginv2.json -o model-pluginv2-rebuild.trt

export TRT_SHIM_OUTPUT_JSON_FILE=$(pwd)/model-pluginv3.json

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtensorrt_shim.so CASE=pluginv3 python main.py

tensorrt_player -j model-pluginv3.json -o model-pluginv3-rebuild.trt
