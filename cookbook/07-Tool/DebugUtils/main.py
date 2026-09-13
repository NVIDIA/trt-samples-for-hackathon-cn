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

from tensorrt_cookbook import print_stacks, run_once, trace_func

@run_once
def init_once():
    print("init_once() called")

@trace_func
def traced_add(a, b):
    c = a + b
    return c

if __name__ == "__main__":

    # No-op on second call
    init_once()
    init_once()

    # Track the function call stack
    traced_add(3, 4)

    # Print the function call stack
    print_stacks()

    print("finish")
