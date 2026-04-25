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

try:
    import nvtripy as tp
except ImportError:
    import tripy as tp

required_symbols = ["compile", "InputInfo", "iota", "gelu", "float32"]
if any(not hasattr(tp, name) for name in required_symbols):
    raise ImportError("This example requires NVIDIA Tripy (`nvtripy`). "
                      "Install it with: python3 -m pip install nvtripy -f "
                      "https://nvidia.github.io/TensorRT-Incubator/packages.html")

def gelu_block(x):
    return tp.gelu(x)

inp = tp.iota(shape=(1, 2), dim=1, dtype=tp.float32).eval()

eager_out = gelu_block(inp)
eager_out.eval()
print("Eager out:", eager_out)

fast_gelu = tp.compile(
    gelu_block,
    args=[tp.InputInfo(shape=(1, 2), dtype=tp.float32)],
)

compiled_out = fast_gelu(inp)
compiled_out.eval()
print("Compiled out:", compiled_out)
