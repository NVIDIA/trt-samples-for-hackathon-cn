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
from tensorrt_cookbook import case_mark, cookbook_path, parse_onnx, TRTWrapperV1

onnx_file = cookbook_path("00-Data", "model", "model-if.onnx")
data = {"x": np.arange(10, dtype=np.int32)}
trt_file = Path("model.trt")

@case_mark
def case_normal():

    tw = TRTWrapperV1()
    parse_onnx(onnx_file, tw.logger, tw.network, tw.builder_config)

    input_tensor = tw.network.get_input(0)
    tw.profile.set_shape(input_tensor.name, [1], data["x"].shape, [data["x"].shape[0] * 2])

    tw.build()
    tw.serialize_engine(trt_file)

    tw.setup(data)
    tw.infer()
    return

if __name__ == "__main__":
    trt_file.unlink(missing_ok=True)

    case_normal()  # TODO: correct this example

    print("Finish")
