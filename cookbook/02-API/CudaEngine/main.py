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

import os

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import APIExcludeSet, TRTWrapperV1, case_mark

shape = [3, 4, 5]
data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
trt_file = "model.trt"

@case_mark
def case_normal():
    tw = TRTWrapperV1()

    tw.network.name = "a_cute_network"
    tensor0 = tw.network.add_input("inputT0", trt.float32, [-1] + shape[1:])
    tw.profile.set_shape(tensor0.name, [1] + shape[1:], shape, [7] + shape[1:])
    tensor1 = tw.network.add_input("inputT1", trt.int32, [3])
    tw.profile.set_shape_input(tensor1.name, [1] + shape[1:], shape, [7] + shape[1:])
    tw.config.add_optimization_profile(tw.profile)

    layer = tw.network.add_shuffle(tensor0)
    layer.set_input(1, tensor1)
    tensor = layer.get_output(0)
    tensor.name = "outputT0"

    tw.build([tensor])

    engine = trt.Runtime(tw.logger).deserialize_cuda_engine(tw.engine_bytes)

    callback_member, callable_member, attribution_member = APIExcludeSet.split_members(engine)
    print(f"\n{'='*64} Members of trt.ICudaEngine:")
    print(f"{len(callback_member):2d} Members to get/set callback classes: {callback_member}")
    print(f"{len(callable_member):2d} Callable methods: {callable_member}")
    print(f"{len(attribution_member):2d} Non-callable attributions: {attribution_member}")

    print(f"{engine.error_recorder = }")  # -> 04-Feature/ErrorRecorder

    print(f"\n{'='*64} Meta data related")
    n_io_tensor = engine.num_io_tensors
    tnl = [engine.get_tensor_name(i) for i in range(n_io_tensor)]  # io_tensor_name_list, tnl for short

    print(f"{engine.name = }")
    print(f"{engine.device_memory_size_v2 = }")
    print(f"{engine.engine_capability = }")
    print(f"{engine.hardware_compatibility_level = }")
    print(f"{engine.num_aux_streams = }")
    print(f"{engine.num_optimization_profiles = }")
    print(f"{engine.refittable = }")
    print(f"{engine.tactic_sources = }")
    print(f"{engine.get_device_memory_size_for_profile_v2(0) = }")

    print(f"{engine.refittable = }")
    print(f"{engine.profiling_verbosity = }")

    print(f"\n{'='*64} Layer related")
    print(f"{engine.num_layers = }")

    print(f"\n{'='*64} Tensor related")
    print(f"{engine.num_io_tensors = }")
    print(f"{[engine.get_tensor_name(i) for i in range(engine.num_io_tensors)] = } <-'tnl' for short")
    print(f"{[engine.get_tensor_mode(i) for i in tnl] = }")
    print(f"{[engine.get_tensor_location(i) for i in tnl] = }")
    print(f"{[engine.get_tensor_shape(i) for i in tnl] = }")
    print(f"{[engine.get_tensor_dtype(i) for i in tnl] = }")
    print(f"{[engine.get_tensor_format(i) for i in tnl] = }")
    print(f"{[engine.get_tensor_format_desc(i) for i in tnl] = }")
    print(f"{[engine.get_tensor_vectorized_dim(i) for i in tnl] = }")  # -> 98-Uncategorized/DataFormat
    print(f"{[engine.get_tensor_components_per_element(i) for i in tnl] = }")  # -> 98-Uncategorized/DataFormat
    print(f"{[engine.get_tensor_bytes_per_component(i) for i in tnl] = }")  # -> 98-Uncategorized/DataFormat
    print(f"{[engine.is_shape_inference_io(i) for i in tnl] = }")
    print(f"{[engine.is_debug_tensor(i) for i in tnl] = }")
    print(f"{engine.get_tensor_profile_shape(tnl[0], 0)  = }, only for input execution tensor")
    print(f"{engine.get_tensor_profile_values(0, tnl[1])  = }, only for input shape tensor")

    print(f"\n{'='*64} Inspector related")
    inspector = engine.create_engine_inspector()  # -> 04-Feature/EngineInspector

    print(f"\n{'='*64} Other APIs")
    engine.create_execution_context()  # Create an execution context from engine in runtime

@case_mark
def case_weight_streaming():
    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tw.config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)  # enable this flag to use weight_streaming

    tensor = tw.network.add_input("inputT0", trt.float32, shape)
    layer = tw.network.add_shuffle(tensor)

    tw.build([layer.get_output(0)])

    engine = trt.Runtime(tw.logger).deserialize_cuda_engine(tw.engine_bytes)

    print(f"{engine.streamable_weights_size = }")
    print(f"{engine.get_weight_streaming_automatic_budget() = }")
    print(f"{engine.weight_streaming_budget_v2 = }")
    print(f"{engine.weight_streaming_scratch_memory_size = }")

@case_mark
def case_serialize():
    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.REFIT)  # Enable this flag to serialize a engine with EXCLUDE_WEIGHTS

    tensor = tw.network.add_input("inputT0", trt.float32, shape)

    w = np.ascontiguousarray(np.random.rand(1, 5, 6).astype(np.float32))  # Build a network with weights
    constant_layer = tw.network.add_constant(w.shape, trt.Weights(w))
    layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, constant_layer.get_output(0), trt.MatrixOperation.NONE)
    tw.build([layer.get_output(0)])

    engine = trt.Runtime(tw.logger).deserialize_cuda_engine(tw.engine_bytes)

    print(f"{engine.create_runtime_config() = }")

    engine_bytes = engine.serialize()
    with open(trt_file, "wb") as f:  # Save normal engine
        f.write(engine_bytes)
    print(f"Size of full engine          : {os.path.getsize(trt_file):8d}B")

    serialize_config = engine.create_serialization_config()

    serialize_config.set_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS)
    engine_bytes = engine.serialize_with_config(serialize_config)
    with open(trt_file, "wb") as f:  # Save engine with config EXCLUDE_WEIGHTS
        f.write(engine_bytes)
    print(f"Size of no-Weight engine     : {os.path.getsize(trt_file):8d}B")
    serialize_config.clear_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS)

    serialize_config.set_flag(trt.SerializationFlag.EXCLUDE_LEAN_RUNTIME)
    engine_bytes = engine.serialize_with_config(serialize_config)
    with open(trt_file, "wb") as f:  # Save engine with config EXCLUDE_LEAN_RUNTIME
        f.write(engine_bytes)
    print(f"Size of no-LeanRuntime engine: {os.path.getsize(trt_file):8d}B")

if __name__ == "__main__":
    case_normal()
    case_weight_streaming()
    case_serialize()

    print("Finish")
"""
APIs not showed:
create_execution_context_without_device_memory  -> deprecated by create_execution_context
device_memory_size                              -> deprecated by device_memory_size_v2
get_device_memory_size_for_profile              -> deprecated by get_device_memory_size_for_profile_v2
has_implicit_batch_dimension                    -> deprecated in TensorRT-10, always return `False`
minimum_weight_streaming_budget                 -> deprecated by weight_streaming_budget_v2
weight_streaming_budget                         -> deprecated by weight_streaming_budget_v2
"""
