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

from cuda.bindings import runtime as cudart

_, n_device = cudart.cudaGetDeviceCount()

for i_device in range(n_device):
    print("=" * 64 + f" Device {i_device}")
    _, info = cudart.cudaGetDeviceProperties(i_device)
    for attr in dir(info):
        if attr.startswith("__"):
            continue
        item = info.__getattribute__(attr)
        if callable(item):  # method
            line = f"{attr}()".ljust(40, " ") + f" = {item()}"
        else:  # variable
            line = f"{attr:40s} = {item}"
        print(line)

print("Finish")
