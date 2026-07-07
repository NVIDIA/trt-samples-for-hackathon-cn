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

import os
import subprocess

def run_cmd(command: str) -> str:
    try:
        return subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as error:
        return f"<failed: {error}>"

if __name__ == "__main__":
    gpu_list = run_cmd("nvidia-smi -L")
    print("=== nvidia-smi -L ===")
    print(gpu_list)

    mig_devices = [line for line in gpu_list.splitlines() if "MIG" in line]
    print(f"\nDetected MIG instances: {len(mig_devices)}")

    if mig_devices:
        print("\nRecommended usage:")
        print("export CUDA_VISIBLE_DEVICES=<MIG-UUID>")
        print("python3 <your_tensorrt_script>.py")
    else:
        print("\nNo MIG instances found. On supported GPUs, create MIG instances first.")

    print(f"\nCurrent CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}")
