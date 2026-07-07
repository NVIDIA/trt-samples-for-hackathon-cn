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

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import case_mark, TRTWrapperV1

trt_file = Path("model.trt")

@case_mark
def case_simple():

    tw = TRTWrapperV1()
    print(f"builder.num_DLA_cores={tw.builder.num_DLA_cores}")
    if tw.builder.num_DLA_cores <= 0:
        print("No DLA core available on current platform")
        return

    tw.builder_config.engine_capability = trt.EngineCapability.DLA_STANDALONE
    tw.builder_config.set_engine_capability(trt.EngineCapability.DLA_STANDALONE)
    tw.builder_config.default_device_type = trt.DeviceType.DLA
    tw.builder_config.DLA_core = 0

    data = np.arange(1 * 1 * 8 * 8, dtype=np.float32).reshape(1, 1, 8, 8)
    input_tensor = tw.network.add_input("inputT0", trt.float32, list(data.shape))

    w = trt.Weights(np.ones((1, 1, 1, 1), dtype=np.float32))
    b = trt.Weights(np.zeros((1, ), dtype=np.float32))
    conv = tw.network.add_convolution_nd(input_tensor, 1, [1, 1], w, b)

    tw.build([conv.get_output(0)])
    tw.serialize_engine(trt_file)

if __name__ == "__main__":
    trt_file.unlink(missing_ok=True)

    case_simple()

    print("Finish")
