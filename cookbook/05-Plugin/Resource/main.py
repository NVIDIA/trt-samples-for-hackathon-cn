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
from tensorrt_cookbook import TRTWrapperV1, case_mark, check_array, get_plugin

@case_mark
def case_resource_share_between_plugins():
    shape = (2, 3, 4)
    input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    trt_file = Path("model-resource-share.trt")
    plugin_file_list = [Path(__file__).parent / "ResourceSharePlugin.so"]

    seed = 2026

    tw = TRTWrapperV1(logger="info", trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:
        plugin_info_dict = {
            "WriterPluginLayer": dict(
                name="ResourceWriter",
                version="1",
                namespace="",
                argument_dict=dict(seed=np.array([seed], dtype=np.int32)),
                number_input_tensor=1,
                number_input_shape_tensor=0,
                plugin_api_version="3",
            ),
            "ReaderPluginLayer": dict(
                name="ResourceReader",
                version="1",
                namespace="",
                argument_dict=dict(),
                number_input_tensor=1,
                number_input_shape_tensor=0,
                plugin_api_version="3",
            ),
        }

        input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
        tw.profile.set_shape(input_tensor.name, [1, 1, 1], shape, shape)

        writer_layer = tw.network.add_plugin_v3([input_tensor], [], get_plugin(plugin_info_dict["WriterPluginLayer"]))
        writer_layer.name = "WriterPluginLayer"

        reader_layer = tw.network.add_plugin_v3([writer_layer.get_output(0)], [], get_plugin(plugin_info_dict["ReaderPluginLayer"]))
        reader_layer.name = "ReaderPluginLayer"

        output_tensor = reader_layer.get_output(0)
        output_tensor.name = "outputT0"

        tw.build([output_tensor])
        tw.serialize_engine(trt_file)

    tw.setup(input_data)
    tw.infer()

    check_array(tw.buffer["outputT0"][0], input_data["inputT0"], True)

if __name__ == "__main__":
    for trt_path in Path(".").glob("*.trt"):
        trt_path.unlink(missing_ok=True)

    case_resource_share_between_plugins()  # Build and run once
    case_resource_share_between_plugins()  # Load and run once

    print("Finish")
