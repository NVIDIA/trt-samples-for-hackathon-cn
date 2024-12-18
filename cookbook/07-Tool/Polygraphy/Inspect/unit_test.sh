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

chmod +x main.sh
./main.sh

polygraphy inspect              --help > Help-inspect.txt
polygraphy inspect model        --help > Help-inspect-model.txt
polygraphy inspect data         --help > Help-inspect-data.txt
polygraphy inspect tactics      --help > Help-inspect-tactics.txt
polygraphy inspect capability   --help > Help-inspect-capability.txt
polygraphy inspect diff-tactics --help > Help-inspect-diff-tactics.txt
polygraphy inspect sparsity     --help > Help-inspect-sparsity.txt

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.json *.log *.onnx *.raw *.trt bad/ good/ polygraphy_capability_dumps/
fi

echo "Finish `basename $(pwd)`"
