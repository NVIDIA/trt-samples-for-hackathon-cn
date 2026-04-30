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

import gc
import os

from tensorrt_cookbook import (
    default_gpus_per_node,
    customized_gc_thresholds,
    get_free_port,
    get_free_ports,
    is_trace_enabled,
    print_stacks,
    release_gc,
    run_once,
    trace_func,
)

@run_once
def init_once():
    print("init_once() called")

@trace_func
def traced_add(a, b):
    c = a + b
    return c

class AttrHolder:
    pass

def main():

    print(f"default_gpus_per_node={default_gpus_per_node()}")

    p0 = get_free_port()
    p_list = get_free_ports(3)
    print(f"free_port={p0}, free_ports={p_list}")

    print(f"default gc threshold = {gc.get_threshold()}")
    with customized_gc_thresholds(gen0_threshold=200):
        print(f"inside customized gc threshold = {gc.get_threshold()}")
    print(f"restored gc threshold = {gc.get_threshold()}")

    init_once()
    init_once()  # no-op on second call

    if is_trace_enabled("TRT_COOKBOOK_TRACE_RANK"):
        value = traced_add(3, 4)
        print(f"traced result = {value}")
    else:
        print("trace disabled, set TRT_COOKBOOK_TRACE_RANK=ALL to enable")

    print_stacks()
    release_gc()
    print("finish")

if __name__ == "__main__":
    # Example:
    # TRT_COOKBOOK_TRACE_RANK=ALL python3 main.py
    os.environ.setdefault("TRT_COOKBOOK_TRACE_RANK", "-1")
    main()
