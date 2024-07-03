#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
#

import tensorrt as trt

output_file = f"result-TesnorRT-{trt.__version__}.log"
ss = ""

def sprint(info):
    global ss
    ss += info + "\n"

def list_api(class_name, level):
    i = 0
    for api_name in dir(class_name):
        if api_name in ["ctypes", "os", "sys", "tensorrt", "warnings"]:  # Skip sub-packages causing infinity recursion
            continue
        if api_name.startswith("__"):  # Skip builtin members and methods
            continue

        if type(class_name).__name__ == "pybind11_type":  # Specialize for pybind11 type
            target = type(class_name).__getattribute__(class_name, api_name)
        else:
            target = class_name.__getattribute__(api_name)

        if type(target).__name__ != "module" and type(target) == type(class_name):  # Skip such cases: type(son) == type(parent)
            continue

        info = f"{'│' * level + '└'}{str(i).zfill(3)}: {api_name}"
        i += 1
        if isinstance(target, (int, float, str, list, dict, tuple, set, bool, bytes)):
            sprint(info + f": {target}")
        elif type(target).__name__ == "pybind11_static_property":
            sprint(info + f": {target.fget(target)}")
        else:
            if hasattr(target, "__call__"):
                info += "()"
            sprint(info)
            if not isinstance(target, (property)):
                list_api(target, level + 1)

    return

list_api(trt, 0)

with open(output_file, "w") as f:
    f.write(ss)

print("Finish")
