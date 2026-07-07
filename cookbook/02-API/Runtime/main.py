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

from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, check_api_coverage, print_enumerated_members

trt_file = Path("model.trt")
data = {"inputT0": np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)}

tw = TRTWrapperV1()

input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
tw.profile.set_shape(input_tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])

layer = tw.network.add_identity(input_tensor)

# Target runtime platform of the engine, used for cross-platform (AOT) engine build.
print_enumerated_members(trt.RuntimePlatform)
# `trt.RuntimePlatform.SAME_AS_BUILD` (default) builds an engine for the current build platform.
# `trt.RuntimePlatform.WINDOWS_AMD64` builds, on a Linux x86-64 host, an engine that will be deserialized and run on Windows AMD64 (must be paired with the version-compatible build flags).
print(f"{tw.builder_config.runtime_platform = }")

tw.build([layer.get_output(0)])

runtime = trt.Runtime(tw.logger)

# Load runtime from library file, lean or dispatch runtime can also be loaded
# runtime = runtime.load_runtime("/usr/lib/x86_64-linux-gnu/libnvinfer_lean.so")

check_api_coverage(runtime)  # Sanity check, unnecessary in normal workflow

print(f"\n{'=' * 64} Usage show")

print(f"{runtime.error_recorder = }")  # Get/set error recorder, refer to 04-Feature/ErrorRecorder
# print(f"{runtime.gpu_allocator = }")  # set GPU allocator, this member has no getter, refer to 04-Feature/GPUAllocator
print(f"{runtime.logger = }")  # Get logger

print(f"{'='*64} Runtime related")
print(f"{runtime.DLA_core =  }")
print(f"{runtime.num_DLA_cores = }")
print(f"{runtime.engine_host_code_allowed = }")
print(f"{runtime.get_plugin_registry() = }")

runtime.max_threads = 16  # Get/set maximum threads that can be used by the Runtime
runtime.temporary_directory = "."  # Get/set temporary directory for runtime, must be used with trt.TempfileControlFlag.ALLOW_TEMPORARY_FILES
runtime.tempfile_control_flags = trt.TempfileControlFlag.ALLOW_TEMPORARY_FILES
# Alternative values of trt.TempfileControlFlag:
# trt.TempfileControlFlag.ALLOW_IN_MEMORY_FILES -> 0, default
# trt.TempfileControlFlag.ALLOW_TEMPORARY_FILES -> 1

tw.engine = runtime.deserialize_cuda_engine(tw.engine_bytes)

print("Finish")
