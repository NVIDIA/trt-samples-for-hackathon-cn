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

function test ()
{
    pushd $1
    chmod +x unit_test.sh
    ./unit_test.sh
    popd
    echo "Finish $1"
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

    test $dir
done

python3 build-README.py

echo "Finish `basename $(pwd)`"
