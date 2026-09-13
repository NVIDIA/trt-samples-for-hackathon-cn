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

from tensorrt_cookbook import TRTWrapperV1, case_mark

so_file = Path(__file__).parent / "CCLPlugin.so"
np.random.seed(31193)

def get_ccl_plugin():
    for creator in trt.get_plugin_registry().plugin_creator_list:
        if creator.name == "CCLPlugin":
            p0 = trt.PluginField("minPixelScore", np.array([0.7], dtype=np.float32), trt.PluginFieldType.FLOAT32)
            p1 = trt.PluginField("minLinkScore", np.array([0.7], dtype=np.float32), trt.PluginFieldType.FLOAT32)
            p2 = trt.PluginField("minArea", np.array([10], dtype=np.int32), trt.PluginFieldType.INT32)
            p3 = trt.PluginField("maxcomponentCount", np.array([65536], dtype=np.int32), trt.PluginFieldType.INT32)
            return creator.create_plugin(creator.name, trt.PluginFieldCollection([p0, p1, p2, p3]))
    return None

@case_mark
def case_simple(input_shape):
    link_shape = input_shape[:1] + [8] + input_shape[1:]
    input_data = {
        "pixelScore": np.random.rand(int(np.prod(input_shape))).astype(np.float32).reshape(input_shape),
        "linkScore": np.random.rand(int(np.prod(link_shape))).astype(np.float32).reshape(link_shape),
    }

    tw = TRTWrapperV1(plugin_file_list=[so_file])
    input_tensor_0 = tw.network.add_input("pixelScore", trt.float32, (-1, -1, -1))
    tw.profile.set_shape(input_tensor_0.name, [1, 1, 1], [2, 384, 640], [4, 768, 1280])
    input_tensor_1 = tw.network.add_input("linkScore", trt.float32, (-1, 8, -1, -1))
    tw.profile.set_shape(input_tensor_1.name, [1, 8, 1, 1], [4, 8, 384, 640], [8, 8, 768, 1280])

    ccl_layer = tw.network.add_plugin_v2([input_tensor_0, input_tensor_1], get_ccl_plugin())

    tw.build([ccl_layer.get_output(0), ccl_layer.get_output(1)])
    tw.setup(input_data)
    tw.infer()

if __name__ == "__main__":
    case_simple([1, 1, 1])
    case_simple([2, 384, 640])
    case_simple([4, 768, 1280])

    print("Finish")
