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

import inspect

import tensorrt as trt
from tensorrt_cookbook import print_enumerated_members

if __name__ == "__main__":
    for name in sorted(dir(trt)):
        obj = getattr(trt, name)
        if inspect.isclass(obj) and hasattr(obj, "__members__"):
            print_enumerated_members(obj)

    print("Finish")
