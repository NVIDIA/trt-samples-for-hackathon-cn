#!/bin/bash

# Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# This software is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and limitations under the License.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COMMON_ROOT="$SCRIPT_DIR"
while [ "$COMMON_ROOT" != "/" ] && [ ! -f "$COMMON_ROOT/tools/unit_test_common.sh" ]; do
    COMMON_ROOT=$(dirname "$COMMON_ROOT")
done

if [ ! -f "$COMMON_ROOT/tools/unit_test_common.sh" ]; then
    echo "Can not find tools/unit_test_common.sh from $SCRIPT_DIR"
    exit 2
fi

source "$COMMON_ROOT/tools/unit_test_common.sh"
trt_bootstrap_runner "$SCRIPT_DIR"
trt_run_case "$SCRIPT_DIR"

echo "Finish `basename $(pwd)`"
