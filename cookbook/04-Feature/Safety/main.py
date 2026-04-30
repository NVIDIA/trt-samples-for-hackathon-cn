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

from pathlib import Path

import tensorrt as trt
from tensorrt_cookbook import case_mark, TRTWrapperV1, load_mnist_network_trt

trt_file = Path("model.trt")

@case_mark
def case_simple():

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    tw.builder_config.set_flag(trt.BuilderFlag.SAFETY_SCOPE)
    tw.builder_config.engine_capability = trt.EngineCapability.SAFETY

    if True:
        # A simple network
        tensor = tw.network.add_input("inputT0", trt.float32, [3, 4, 5])
        layer = tw.network.add_identity(tensor)
        tw.build([layer.get_output(0)])
    else:
        # A larger network
        load_mnist_network_trt(tw, b_dynamic_shape=False)
        tw.build()

    try:
        tw.runtime = trt.Runtime(tw.logger)
        tw.engine = tw.runtime.deserialize_cuda_engine(tw.engine_bytes)
        print(f"Safety engine capability = {tw.engine.engine_capability}")
    except Exception as e:
        print(f"Failed to create a safety engine: {e}")

if __name__ == "__main__":
    trt_file.unlink(missing_ok=True)

    case_simple()

    print("Finish")
