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

def case_compute_capability():
    print("\n" + "-" * 64)
    print("Compute capability APIs")
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

def case_builder_flag_require_user_allocation():
    print("\n" + "-" * 64)
    print("BuilderFlag.REQUIRE_USER_ALLOCATION")
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    builder_config = builder.create_builder_config()
    builder_config.set_flag(trt.BuilderFlag.REQUIRE_USER_ALLOCATION)
    print(f"builder_config.get_flag(REQUIRE_USER_ALLOCATION) = {builder_config.get_flag(trt.BuilderFlag.REQUIRE_USER_ALLOCATION)}")

def case_runtime_validity_and_runtime_config(engine_bytes: bytes):
    print("\n" + "-" * 64)
    print("Runtime validity / runtime builder_config / runtime cache")

    logger = trt.Logger(trt.Logger.ERROR)
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

def case_is_stream_capturable(engine_bytes: bytes):
    print("\n" + "-" * 64)
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

def case_enums():
    print("\n" + "-" * 64)
    print("RTX-only enums")
    print(f"ComputeCapability.CURRENT = {trt.ComputeCapability.CURRENT}")
    print(f"CudaGraphStrategy.WHOLE_GRAPH_CAPTURE = {trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE}")
    print(f"DynamicShapesKernelSpecializationStrategy.EAGER = {trt.DynamicShapesKernelSpecializationStrategy.EAGER}")
    print(f"EngineValidity.VALID = {trt.EngineValidity.VALID}")
    print(f"EngineInvalidityDiagnostics.VERSION_MISMATCH = {trt.EngineInvalidityDiagnostics.VERSION_MISMATCH}")

if __name__ == "__main__":
    case_compute_capability()
    case_builder_flag_require_user_allocation()

    logger = trt.Logger(trt.Logger.ERROR)
    engine_bytes = build_engine_bytes(logger)
    case_runtime_validity_and_runtime_config(engine_bytes)
    case_is_stream_capturable(engine_bytes)
    case_enums()

    print("Finish")
