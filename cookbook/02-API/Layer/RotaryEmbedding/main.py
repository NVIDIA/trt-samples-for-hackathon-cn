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

import numpy as np
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_cast, check_api_coverage

@case_mark
def case_simple():
    b, n, s, h = 1, 2, 4, 8
    max_position = 16
    data = {
        "input": np.arange(b * n * s * h, dtype=np.float32).reshape(b, n, s, h) / 32,
        "cos_cache": np.cos(np.linspace(0, 1, max_position * (h // 2), dtype=np.float32)).reshape(max_position, h // 2),
        "sin_cache": np.sin(np.linspace(0, 1, max_position * (h // 2), dtype=np.float32)).reshape(max_position, h // 2),
        "position_ids": np.tile(np.arange(s, dtype=np.int64), (b, 1)),
    }

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("input", datatype_cast(data["input"].dtype, "trt"), data["input"].shape)
    cos_cache = tw.network.add_input("cos_cache", datatype_cast(data["cos_cache"].dtype, "trt"), data["cos_cache"].shape)
    sin_cache = tw.network.add_input("sin_cache", datatype_cast(data["sin_cache"].dtype, "trt"), data["sin_cache"].shape)
    position_ids = tw.network.add_input("position_ids", datatype_cast(data["position_ids"].dtype, "trt"), data["position_ids"].shape)

    layer = tw.network.add_rotary_embedding(tensor, cos_cache, sin_cache, False, 0)
    # Input: input: T[b, d, s, h], cos_cache: T[b, s, h/2] or T[max_position_id+1, h/2], sin_cache: same shape as cos_cache, position_ids (optional): M[b, s]
    # Output: T[b, d, s, h]
    # Data Type: T in [float16, float32, bfloat16], M (position_ids) is int64
    # Shape: input and output share shape [b, d, s, h]; cos_cache/sin_cache last dim becomes rotary_embedding_dim/2 when rotary_embedding_dim != 0
    layer.set_input(3, position_ids)
    layer.interleaved = False  # Reset later
    layer.rotary_embedding_dim = 0  # Reset later

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    output_tensor = layer.get_output(0)
    output_tensor.name = "output"
    if not tw.build([output_tensor]):
        print("Fail building rotary-embedding engine")
        return
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Basic RoPE (Rotary Position Embedding) example
    case_simple()

    print("Finish")
