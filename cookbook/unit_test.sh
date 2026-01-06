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

export TRT_COOKBOOK_PATH=$(pwd)

function test ()
{
    pushd $1
    chmod +x unit_test.sh
    ./unit_test.sh
    popd
    echo "Finish $1"
}

EXCLUDE_LIST=\
"""
09-TRT-LLM/
99-Todo/
build/
dist/
include/
tensorrt_cookbook/
tensorrt_cookbook.egg-info/
"""

SKIP_LIST=\
"""
"""

BACKUP_LIST=\
"""
00-Data/
01-SimpleDemo/
02-API/
03-Workflow/
04-Feature/
05-Plugin/
06-DLFrameworkTRT/
07-Tool/
08-Advance/
09-TRTLLM/
98-Uncategorized/
"""

for dir in */;
do
    if echo $EXCLUDE_LIST | grep -q $dir; then
        continue
    fi
    # Skip when the tests fail at somewhere in the half way
    if echo $SKIP_LIST | grep -q $dir; then
        continue
    fi

    test $dir
done

echo "Finish ALL tests"
