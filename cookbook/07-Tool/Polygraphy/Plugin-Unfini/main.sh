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
rm -rf *.log *.onnx *.so *.yaml
#clear

# 00-Get ONNX model
export MODEL_ADDSCALAR=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx

pushd $TRT_COOKBOOK_PATH/05-Plugin/BasicExample
make clean
make all -j
popd
cp $TRT_COOKBOOK_PATH/05-Plugin/BasicExample/AddScalarPlugin.so .

#01-?
polygraphy plugin list $MODEL_ADDSCALAR \
    --plugin-dir . \
    > reuslt-01.log 2>&1

#02-?
polygraphy plugin match $MODEL_ADDSCALAR \
    --plugin-dir . \
    > result-02.log 2>&1

#03-?
polygraphy plugin replace $MODEL_ADDSCALAR \
    --plugin-dir . \
    -o model-custom-op-RE.onnx \
    > result-03.log 2>&1

echo "Finish"
