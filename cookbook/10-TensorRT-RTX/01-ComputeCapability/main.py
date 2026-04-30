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

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
builder_config = builder.create_builder_config()

print(f"{builder_config.num_compute_capabilities = }")

if builder_config.num_compute_capabilities <= 0:
    print("No compute-capability slots available in current runtime, skip setting.")
    print("Finish")
    raise SystemExit(0)

# TensorRT-RTX specific API
try:
    builder_config.set_compute_capability(trt.ComputeCapability.CURRENT, 0)
except TypeError:
    builder_config.set_compute_capability(0, trt.ComputeCapability.CURRENT)

try:
    current_cc = builder_config.get_compute_capability(0)
except TypeError:
    current_cc = builder_config.get_compute_capability()

print(f"Configured compute capability = {current_cc}")
print("Finish")
