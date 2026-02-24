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

function test ()
{
    trt_test_subdir "$1"
}

SKIP_LIST=\
"""
APIs/
APIs-V2-deprecated/
"""

for dir in */;
do
    if echo $SKIP_LIST | grep -q $dir; then
        continue
    fi

    test "$dir"
done


echo "Finish `basename $(pwd)`"
