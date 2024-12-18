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

nsys                --help > Help.txt
nsys analyze        --help > Help-analyze.txt
nsys cancel         --help > Help-cancel.txt
nsys export         --help > Help-export.txt
nsys profile        --help > Help-profile.txt
nsys launch         --help > Help-launch.txt
nsys stop           --help > Help-stop.txt
nsys service        --help > Help-service.txt
nsys stats          --help > Help-stats.txt
nsys shutdown       --help > Help-shutdown.txt
nsys sessions list  --help > Help-sessions-list.txt
nsys recipe         --help > Help-recipe.txt
nsys nvprof         --help > Help-nvprof.txt

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.log *.onnx *.nsys-rep *.qdrep *.qdrep-nsys *.trt
fi

echo "Finish `basename $(pwd)`"
