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
#

from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark, check_array, get_plugin

@case_mark
def case_simple():
    scalar = 1.0
    shape = (3, 4, 5)
    input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    trt_file = Path("model.trt")
    plugin_file_list = [Path(__file__).parent / "AddScalarPlugin.so"]

    def add_scalar_cpu(buffer, scalar):
        return {"outputT0": buffer["inputT0"] + scalar}

    tw = TRTWrapperV1(logger="info", trt_file=trt_file, plugin_file_list=plugin_file_list)  # Use log level "info" to show log inside plugin
    if tw.engine_bytes is None:  # Create engine from scratch

        plugin_info_dict = {
            "AddScalarPluginLayer": dict(
                name="AddScalar",
                version="1",
                namespace="",
                argument_dict=dict(scalar=np.array([scalar], dtype=np.float32)),
                number_input_tensor=1,
                number_input_shape_tensor=0,
                plugin_api_version="3",
            )
        }

        input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
        tw.profile.set_shape(input_tensor.name, [1, 1, 1], shape, shape)
        tw.config.add_optimization_profile(tw.profile)

        layer = tw.network.add_plugin_v3([input_tensor], [], get_plugin(plugin_info_dict["AddScalarPluginLayer"]))
        layer.name = "AddScalarPluginLayer"
        tensor = layer.get_output(0)
        tensor.name = "outputT0"

        tw.build([tensor])
        tw.serialize_engine(trt_file)

    tw.setup(input_data)
    tw.infer()

    output_cpu = add_scalar_cpu(input_data, scalar)

    check_array(tw.buffer["outputT0"][0], output_cpu["outputT0"], True)

if __name__ == "__main__":
    for trt_path in Path(".").glob("*.trt"):
        trt_path.unlink(missing_ok=True)
    # A simple case of using pluginv3 layer
    case_simple()  # Build engine and plugin to do inference
    case_simple()  # Load engine and plugin to do inference

    print("Finish")
