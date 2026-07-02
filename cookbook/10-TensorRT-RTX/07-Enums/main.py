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

import tensorrt_rtx as trt

def print_enum(enum_type, title: str):
    print("\n" + title)
    members = getattr(enum_type, "__members__", None)
    if isinstance(members, dict):
        for name, item in members.items():
            print(f"{name:40s} -> {item.value}")
        return

    for name in sorted(dir(enum_type)):
        if name.startswith("_"):
            continue
        value = getattr(enum_type, name)
        if isinstance(value, enum_type):
            print(f"{name:40s} -> {value.value}")

if __name__ == "__main__":
    print_enum(trt.ComputeCapability, "ComputeCapability")
    print_enum(trt.CudaGraphStrategy, "CudaGraphStrategy")
    print_enum(trt.DynamicShapesKernelSpecializationStrategy, "DynamicShapesKernelSpecializationStrategy")
    print_enum(trt.EngineValidity, "EngineValidity")
    print_enum(trt.EngineInvalidityDiagnostics, "EngineInvalidityDiagnostics")

    print("\nFinish")
