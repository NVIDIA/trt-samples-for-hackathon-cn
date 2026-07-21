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

# Query the GPU device information and current clock/power state.
#
# This is the cookbook re-implementation of `trt-engine-explorer`'s
# `utils/device_info.py` and the read-only parts of `utils/config_gpu.py`,
# using pynvml instead of pycuda. It REQUIRES A GPU and the `pynvml` package.
#
# Engine profiling is most reproducible when the GPU clocks are locked to a
# fixed frequency (config_gpu does this). Locking clocks needs root privileges,
# so this example only *reads* the current and max clocks and reports whether
# the GPU is currently running below its maximum (a sign it may be throttling).

import json
from pathlib import Path

from tensorrt_cookbook import case_mark, get_max_clocks, query_device_info, sample_gpu_state

out_path = Path(__file__).parent
device_json = out_path / "device_info.json"
gpu_id = 0

@case_mark
def case_device_info():
    """Print and dump the metadata of every visible GPU."""
    devices = query_device_info()
    print(f"Found {len(devices)} device(s):")
    for i, dev in enumerate(devices):
        print(f"  Device {i}: {dev['Name']}")
        print(f"    Total memory : {dev['TotalMemory'] / 1024**3:.1f} GiB")
        print(f"    Max SM clock : {dev['MaxSMClockMHz']} MHz")
        print(f"    Max mem clock: {dev['MaxMemClockMHz']} MHz")

    with open(device_json, "w") as f:
        json.dump(devices, f, indent=2)
    print(f"Wrote {device_json.name}")

@case_mark
def case_gpu_state():
    """Sample the current GPU state and compare clocks against the maximum."""
    state = sample_gpu_state(gpu_id)
    max_sm, max_mem = get_max_clocks(gpu_id)

    print(f"Current GPU {gpu_id} state:")
    for k, v in state.items():
        print(f"    {k}: {v}")

    print(f"\nMax clocks: sm={max_sm} MHz, mem={max_mem} MHz")
    if state["sm_clock_mhz"] < max_sm:
        print("Note: the SM clock is below its maximum. For reproducible profiling, "
              "lock the clocks (needs root):")
        print(f"    sudo nvidia-smi -i {gpu_id} --lock-gpu-clocks={max_sm},{max_sm}")
    else:
        print("The SM clock is at its maximum.")

if __name__ == "__main__":
    case_device_info()
    case_gpu_state()

    print("Finish")
