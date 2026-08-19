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
import numpy as np
import tensorrt as trt
from pathlib import Path

trt_file = Path("model.trt")
input_tensor_name = "inputT0"
data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    if os.path.isfile(trt_file):
        with open(trt_file, "rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Fail getting serialized engine")
            return
        print("Succeed getting serialized engine")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        builder_config = builder.create_builder_config()
        builder_config.set_flag(trt.BuilderFlag.SAFETY_SCOPE)  # use Safety mode
        builder_config.engine_capability = trt.EngineCapability.SAFETY  # use Safety mode

        inputTensor = network.add_input("inputT0", trt.float32, [3, 4, 5])  # only Explicit Batch + Static Shape is supported in safety mode

        identityLayer = network.add_identity(inputTensor)
        network.mark_output(identityLayer.get_output(0))

        engineString = builder.build_serialized_network(network, builder_config)
        if engineString == None:
            print("Fail building serialized engine")
            return
        print("Succeed building serialized engine")
        with open(trt_file, "wb") as f:
            f.write(engineString)
            print("Succeed saving .trt file")

    # Error like:
    # [TRT] [E] IRuntime::deserializeCudaEngine: Error Code 1: Serialization (Serialization assertion header.magicTag == kEXPECTED_MAGIC_TAG failed.Trying to load an engine created with incompatible serialization version (1297697870 != 1953657958). Check that the engine was not created using safety runtime, same OS was used and version compatibility parameters were set accordingly and that it is a TRT engine file. In throwUnlessHeaderOk at /_src/runtime/dispatch/runtime.cpp:42)
    # Reason:
    # Engine built with `EngineCapability.SAFETY` / `BuilderFlag.SAFETY_SCOPE`
    # can only be deserialized and executed by the TensorRT *safety runtime*
    # (shipped with NVIDIA DRIVE OS / the safety-certified package), not by the
    # standard datacenter TensorRT runtime used in this cookbook.
    print("Deserializing engine...")
    try:
        trt.Runtime(logger).deserialize_cuda_engine(engineString)
    except Exception:
        pass
    return

if __name__ == "__main__":
    trt_file.unlink(missing_ok=True)

    run()

    print("Finish")
