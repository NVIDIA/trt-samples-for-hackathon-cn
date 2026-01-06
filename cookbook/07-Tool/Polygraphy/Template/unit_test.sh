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

chmod +x main.sh
./main.sh

polygraphy template             --help > Help-template.txt
polygraphy template trt-network --help > Help-template-trt-network.txt
polygraphy template trt-config  --help > Help-template-trt-config.txt
polygraphy template onnx-gs     --help > Help-template-onnx-gs.txt

if [ "${TRT_COOKBOOK_CLEAN-}" ]; then
    rm -rf *.log *.onnx *.trt modify_config.py modify_network.py modify_onnx.py
fi

echo "Finish `basename $(pwd)`"
