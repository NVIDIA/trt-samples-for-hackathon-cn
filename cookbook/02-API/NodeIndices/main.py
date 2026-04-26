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
#

# from pathlib import Path
import tensorrt as trt

# from tensorrt_cookbook import APIExcludeSet, grep_used_members

node_indices = trt.NodeIndices([1, 3, 5])
# public_member = APIExcludeSet.analyze_public_members(node_indices, b_print=True)
# grep_used_members(Path(__file__), public_member)

try:
    # Always rasing RecursionError: maximum recursion depth exceeded while calling a Python object
    node_indices.count()
    node_indices.append(7)
    node_indices.insert(1, 2)
    node_indices.extend([9, 11])
    node_indices.remove(3)
    node_indices.pop()
    node_indices.clear()
except RecursionError:
    print("RecursionError")

print("Finish")
