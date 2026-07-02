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

def build_engine_bytes(logger: trt.Logger) -> bytes:
    builder = trt.Builder(logger)
    builder_config = builder.create_builder_config()
    network = builder.create_network()
    input_tensor = network.add_input("inputT0", trt.float32, [3, 4, 5])
    layer = network.add_identity(input_tensor)
    network.mark_output(layer.get_output(0))
    engine_bytes = builder.build_serialized_network(network, builder_config)
    if engine_bytes is None:
        raise RuntimeError("Fail building engine bytes")
    return engine_bytes

if __name__ == "__main__":
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(build_engine_bytes(logger))
    if engine is None:
        raise RuntimeError("Fail deserializing engine")

    runtime_config = engine.create_runtime_config()

    runtime_cache = runtime_config.create_runtime_cache()  # TensorRT-RTX specific API
    blob = runtime_cache.serialize()  # IRuntimeCache API
    print(f"Serialized runtime cache size = {blob.nbytes} bytes")

    restored_runtime_cache = runtime_config.create_runtime_cache()
    restored_runtime_cache.deserialize(blob)  # IRuntimeCache API

    runtime_config.set_runtime_cache(restored_runtime_cache)  # TensorRT-RTX specific API
    print(f"runtime_config.get_runtime_cache() = {runtime_config.get_runtime_cache()}")

    restored_runtime_cache.reset()  # IRuntimeCache API

    print("Finish")
