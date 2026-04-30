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

import tensorrt_rtx as trt

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
builder_config = builder.create_builder_config()
network = builder.create_network()

input_tensor = network.add_input("inputT0", trt.float32, [3, 4, 5])
layer = network.add_identity(input_tensor)
network.mark_output(layer.get_output(0))

# TensorRT-RTX specific BuilderFlag
builder_config.set_flag(trt.BuilderFlag.REQUIRE_USER_ALLOCATION)
print(f"builder_config.get_flag(REQUIRE_USER_ALLOCATION) = {builder_config.get_flag(trt.BuilderFlag.REQUIRE_USER_ALLOCATION)}")

try:
    engine_bytes = builder.build_serialized_network(network, builder_config)
    print(f"Build result with REQUIRE_USER_ALLOCATION: {'Success' if engine_bytes is not None else 'Failed'}")
except Exception as error:
    print(f"Build raised exception: {error}")

print("Finish")
