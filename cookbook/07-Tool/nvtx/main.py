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
import nvtx
from cuda.bindings import runtime as cudart
import numpy as np
from tensorrt_cookbook import (
    TRTWrapperV1,
    enable_gc_nvtx_profiling,
    load_mnist_network_trt,
)

trt_file = Path("model.trt")
data = {"x": np.arange(1 * 1 * 28 * 28, dtype=np.float32).reshape(1, 1, 28, 28)}

tw = TRTWrapperV1()

load_mnist_network_trt(tw)

tw.build()

tw.setup(data)

# Optional, enable GC->NVTX callback if `TRT_COOKBOOK_PROFILE_RECORD_GC=1`
enable_gc_nvtx_profiling()

# Single mark example
nvtx.mark("build_done", color="black", domain="NVTX-cookbook", category="setup")

# Range example with context manager
with nvtx.annotate(f"infer", color="yellow", domain="NVTX-cookbook", category="multi-steps"):

    # Three equivalent ways to mark the inference step
    cudart.cudaDeviceSynchronize()
    for i in range(10):
        with nvtx.annotate("enqueue", color="red", domain="NVTX-cookbook", category="step-red"):
            tw.context.execute_async_v3(0)
    cudart.cudaDeviceSynchronize()

    cudart.cudaDeviceSynchronize()
    for i in range(10):
        nvtx.push_range("enqueue", color="green", domain="NVTX-cookbook", category="step-green")
        tw.context.execute_async_v3(0)
        nvtx.pop_range()
    cudart.cudaDeviceSynchronize()

    # start_range / end_range can be set in different threads, comparing to push_range / pop_range which must be in the same thread
    cudart.cudaDeviceSynchronize()
    for i in range(10):
        range_id = nvtx.start_range("enqueue", color="blue", domain="NVTX-cookbook", category="step-blue")
        tw.context.execute_async_v3(0)
        nvtx.end_range(range_id)
    cudart.cudaDeviceSynchronize()

print("Finish")
