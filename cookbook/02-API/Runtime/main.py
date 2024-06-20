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

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1

trt_file = Path("model.trt")
data = {"inputT0": np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)}

tw = TRTWrapperV1()

input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
tw.profile.set_shape(input_tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
tw.config.add_optimization_profile(tw.profile)

layer = tw.network.add_identity(input_tensor)

tw.build([layer.get_output(0)])

runtime = trt.Runtime(tw.logger)

print("================================================================ Runtime related")
print(f"{runtime.logger = }")
print(f"{runtime.DLA_core =  }")
print(f"{runtime.num_DLA_cores = }")
print(f"{runtime.engine_host_code_allowed = }")
print(f"{runtime.error_recorder = }")  # -> 04-Feature/ErrorRecorder
print(f"{runtime.get_plugin_registry() = }")
#print(f"{runtime.gpu_allocator = }")  # unreadable, -> 04-Feature/GPUAllocator

runtime.max_threads = 16  # The maximum thread that can be used by the Runtime
#tw.runtime.temporary_directory = "."
tempfile_control_flags = trt.TempfileControlFlag.ALLOW_TEMPORARY_FILES
# Alternative values of trt.TempfileControlFlag:
# trt.TempfileControlFlag.ALLOW_IN_MEMORY_FILES -> 0, default
# trt.TempfileControlFlag.ALLOW_TEMPORARY_FILES -> 1

tw.engine = runtime.deserialize_cuda_engine(tw.engine_bytes)

print("Finish")
"""
APIs not showed:
load_runtime
"""
