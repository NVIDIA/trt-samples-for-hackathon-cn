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
from tensorrt_cookbook import print_enumerated_members

def case_normal(logger: trt.Logger = None):

    print("\n" + "-" * 64 + " Build phase")
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    builder_config = builder.create_builder_config()

    print(f"{builder_config.num_compute_capabilities = }")
    if builder_config.num_compute_capabilities <= 0:
        print("No compute-capability slots available in current runtime, skip setting.")
        return

    target_cc = trt.ComputeCapability.CURRENT
    try:
        builder_config.set_compute_capability(target_cc, 0)
    except TypeError:
        builder_config.set_compute_capability(0, target_cc)
    try:
        print(f"get_compute_capability(0) = {builder_config.get_compute_capability(0)}")
    except TypeError:
        print(f"get_compute_capability() = {builder_config.get_compute_capability()}")

    network = builder.create_network()
    input_tensor = network.add_input("inputT0", trt.float32, [3, 4, 5])
    layer = network.add_identity(input_tensor)
    network.mark_output(layer.get_output(0))
    engine_bytes = builder.build_serialized_network(network, builder_config)
    if engine_bytes is None:
        raise RuntimeError("Fail building engine bytes")

    print("\n" + "-" * 64 + " Runtime phase")

    runtime = trt.Runtime(logger)
    print(f"{runtime.engine_header_size = }")

    validity_result = runtime.get_engine_validity(engine_bytes)
    print(f"runtime.get_engine_validity(...) = {validity_result}")

    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("Fail deserializing engine")

    runtime_config = engine.create_runtime_config()
    runtime_config.cuda_graph_strategy = trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE
    runtime_config.dynamic_shapes_kernel_specialization_strategy = trt.DynamicShapesKernelSpecializationStrategy.EAGER

    print(f"{runtime_config.cuda_graph_strategy = }")
    print(f"{runtime_config.dynamic_shapes_kernel_specialization_strategy = }")

    runtime_cache = runtime_config.create_runtime_cache()
    cache_blob = runtime_cache.serialize()
    restored_runtime_cache = runtime_config.create_runtime_cache()
    restored_runtime_cache.deserialize(cache_blob)
    runtime_config.set_runtime_cache(restored_runtime_cache)
    print(f"runtime_config.get_runtime_cache() = {runtime_config.get_runtime_cache()}")
    restored_runtime_cache.reset()

    print("\n" + "-" * 64 + " ExecutionContext.is_stream_capturable")
    print("ExecutionContext.is_stream_capturable")
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("Fail deserializing engine")

    context = engine.create_execution_context()
    try:
        result = context.is_stream_capturable()
    except TypeError:
        result = context.is_stream_capturable(0)
    print(f"is_stream_capturable = {result}")

if __name__ == "__main__":

    case_normal()

    print_enumerated_members(trt.ComputeCapability)
    print_enumerated_members(trt.CudaGraphStrategy)
    print_enumerated_members(trt.DynamicShapesKernelSpecializationStrategy)
    print_enumerated_members(trt.EngineInvalidityDiagnostics)
    print_enumerated_members(trt.EngineValidity)

    print("Finish")
