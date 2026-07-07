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

import os
from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark, print_enumerated_members

trt_file = Path("model.trt")
data = {"inputT0": np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)}

def build_engine() -> trt.ICudaEngine:
    tw = TRTWrapperV1()
    input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
    tw.profile.set_shape(input_tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])

    w = np.ascontiguousarray(np.random.rand(1, 5, 6).astype(np.float32))  # Build a network with weights
    constant_layer = tw.network.add_constant(w.shape, trt.Weights(w))
    layer = tw.network.add_matrix_multiply(input_tensor, trt.MatrixOperation.NONE, constant_layer.get_output(0), trt.MatrixOperation.NONE)
    tw.builder_config.set_flag(trt.BuilderFlag.REFIT)  # Needed by SerializationFlag.INCLUDE_REFIT
    tw.build([layer.get_output(0)])

    engine = trt.Runtime(tw.logger).deserialize_cuda_engine(tw.engine_bytes)
    return engine

@case_mark
def case_serialization_config():
    engine = build_engine()

    # `engine.create_serialization_config()` returns a `trt.ISerializationConfig`, which controls
    # what is written into the serialized engine (e.g. whether to strip weights or the lean runtime).
    serialize_config: trt.ISerializationConfig = engine.create_serialization_config()
    print(f"{serialize_config = }")
    print_enumerated_members(trt.SerializationFlag)

    # Save a full engine as a baseline
    engine_bytes = engine.serialize()
    with open(trt_file, "wb") as f:
        f.write(engine_bytes)
    print(f"Size of full engine          : {os.path.getsize(trt_file):8d}B")

    # Toggle each serialization flag on the `trt.ISerializationConfig` and serialize with it
    serialize_config.set_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS)
    print(f"{serialize_config.get_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS) = }")
    engine_bytes = engine.serialize_with_config(serialize_config)
    with open(trt_file, "wb") as f:
        f.write(engine_bytes)
    print(f"Size of no-Weight engine     : {os.path.getsize(trt_file):8d}B")
    serialize_config.clear_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS)

    serialize_config.set_flag(trt.SerializationFlag.EXCLUDE_LEAN_RUNTIME)
    engine_bytes = engine.serialize_with_config(serialize_config)
    with open(trt_file, "wb") as f:
        f.write(engine_bytes)
    print(f"Size of no-LeanRuntime engine: {os.path.getsize(trt_file):8d}B")
    serialize_config.clear_flag(trt.SerializationFlag.EXCLUDE_LEAN_RUNTIME)

    serialize_config.set_flag(trt.SerializationFlag.INCLUDE_REFIT)
    engine_bytes = engine.serialize_with_config(serialize_config)
    with open(trt_file, "wb") as f:
        f.write(engine_bytes)
    print(f"Size of refitable engine     : {os.path.getsize(trt_file):8d}B")
    serialize_config.clear_flag(trt.SerializationFlag.INCLUDE_REFIT)

if __name__ == "__main__":
    trt_file.unlink(missing_ok=True)

    case_serialization_config()

    print("Finish")
