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

import torch

device_count = torch.cuda.device_count()
print(f"CUDA device count: {device_count}")

for device_index in range(device_count):
    print("=" * 64 + f" Device {device_index}")
    properties = torch.cuda.get_device_properties(device_index)
    attribute_name_list = sorted(name for name in dir(properties) if not name.startswith("_")  # and not callable(getattr(properties, name))
                                 )

    for name in attribute_name_list:
        value = getattr(properties, name)

        if callable(value):  # method
            line = f"{name}()".ljust(40, " ") + f" = {value()}"
        else:  # variable
            line = f"{name:40s} = {value}"
        print(line)

print("Finish")
