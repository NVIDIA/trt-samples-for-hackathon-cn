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

import types

import tensorrt as trt

output_file = f"result-TesnorRT-{trt.__version__}.log"
ss = "trt\n"

def add_ss(info):
    global ss, tt
    ss += info + "\n"

def list_api(class_name, level):
    #n_output = 0
    for i, api_name in enumerate(dir(class_name)):
        if api_name in ["ctypes", "os", "sys", "tensorrt", "warnings"] or \
            api_name.startswith("__"):
            # 1. Sub-packages causing infinity recursion
            # 2. Builtin members and methods
            continue

        target = getattr(class_name, api_name)
        mark = "└" if i == len(dir(class_name)) - 1 else "├"
        info = f"{'│' * level}{mark}─ {api_name}"

        if type(target).__name__ != "module" and type(target) == type(class_name) or \
            isinstance(target, str) and str(class_name).endswith(target):
            # 1. Such cases: type(son) == type(parent)
            # 2. Enumerate values
            continue
        elif isinstance(target, (int, float, str, list, dict, tuple, set, bool, bytes)):
            add_ss(info + f": {target}")
        elif type(target).__name__ in ["pybind11_static_property"]:  # specialization of TRT
            add_ss(info + f": {target.fget(target)}")
        else:
            if isinstance(target, (types.FunctionType, types.BuiltinMethodType, types.BuiltinMethodType)):
                info += "()"
            add_ss(info)
            if not isinstance(target, (property)):
                list_api(target, level + 1)

    return

list_api(trt, 0)

with open(output_file, "w") as f:
    f.write(ss)

print("Finish")
