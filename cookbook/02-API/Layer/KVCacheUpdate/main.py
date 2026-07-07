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
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_cast, print_enumerated_members, check_api_coverage

@case_mark
def case_simple():
    b, n, s_max, s_new, h = 1, 2, 8, 2, 8
    data = {
        "cache": np.zeros((b, n, s_max, h), dtype=np.float16),
        "update": np.arange(b * n * s_new * h, dtype=np.float16).reshape(b, n, s_new, h) / 16,
        "write_indices": np.array([0], dtype=np.int32),
    }

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    cache = tw.network.add_input("cache", datatype_cast(data["cache"].dtype, "trt"), data["cache"].shape)
    update = tw.network.add_input("update", datatype_cast(data["update"].dtype, "trt"), data["update"].shape)
    write_indices = tw.network.add_input("write_indices", datatype_cast(data["write_indices"].dtype, "trt"), data["write_indices"].shape)

    layer = tw.network.add_kv_cache_update(cache, update, write_indices, trt.KVCacheMode.LINEAR)
    # Input: cache [b, d, s_max, h] (T: float32/float16/bfloat16, must be a network input with static s_max);
    #        update [b, d, s, h] (T: float32/float16/bfloat16, s <= s_max);
    #        writeIndices [b] (M: int32/int64, writeIndices[i] + s <= s_max)
    # Outputs: output [b, d, s_max, h] (T), shares device memory address with cache input (in-place update)
    # Data type: T in {float32, float16, bfloat16}; M in {int32, int64}
    # Shape: cache and output: [b, d, s_max, h]; update: [b, d, s, h] where s <= s_max; writeIndices: [b]

    if layer is None:
        print("`add_kv_cache_update` failed. Check TensorRT version and strongly-typed network setting.")
        return

    output_tensor = layer.get_output(0)
    output_tensor.name = "cache_out"

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    if not tw.build([output_tensor]):
        print("Fail building kv-cache-update engine")
        return

    tw.setup(data)

    aliased_input_name = tw.engine.get_aliased_input_tensor("cache_out")
    print(f"{aliased_input_name = }")
    if aliased_input_name is None:
        print("No aliased input found; skip inference.")
        return

    tw.buffer["cache_out"][1] = tw.buffer[aliased_input_name][1]
    tw.context.set_tensor_address("cache_out", tw.buffer["cache_out"][1])
    tw.infer()

if __name__ == "__main__":
    # Basic KV cache update example
    case_simple()  # TODO: Add more cases

    print_enumerated_members(trt.KVCacheMode)

    print("Finish")
