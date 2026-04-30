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

clear && nvidia-smi

# docker system prune -af &

VERSION="${1:-26.03}"

docker run \
    -it \
    --gpus $(nvidia-smi -L | wc -l) \
    --shm-size 32G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name tensorrt-cookbook-${VERSION} \
    -v /home/wili/work:/work \
    -v /home/wili:/wili \
    nvcr.io/nvidia/pytorch:${VERSION}-py3 \
    /bin/bash
