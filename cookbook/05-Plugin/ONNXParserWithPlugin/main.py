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

model_path = cookbook_path("00-Data", "model")
onnx_file = model_path / "model-addscalar.onnx"
data = {"inputT0": np.ones([4], dtype=np.float32)}
trt_file = Path("model.trt")
plugin_file_list = [Path(__file__).parent / "AddScalarPlugin.so"]

@case_mark
def case_normal():
    tw = TRTWrapperV1(trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:  # Create engine from scratch

        parse_onnx(onnx_file, tw.logger, tw.network, tw.builder_config)

        tensor = tw.network.get_input(0)
        tw.profile.set_shape(tensor.name, data["inputT0"].shape, data["inputT0"].shape, data["inputT0"].shape)

        tw.build()
        tw.serialize_engine(trt_file)

    tw.setup(data)
    tw.infer()
    return

if __name__ == "__main__":
    for trt_path in Path(".").glob("*.trt"):
        trt_path.unlink(missing_ok=True)
    case_normal()
    case_normal()

    print("Finish")
