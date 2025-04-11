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
from pathlib import Path

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1, check_array

shape = [3, 4, 5]
input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
trt_file = Path("model.trt")
plugin_file_list = [Path(__file__).parent / "IdentityPlugin.so"]

def identity_cpu(buffer):
    return {"outputT0": buffer["inputT0"]}

def getIdentityPlugin():
    name = "Identity"
    plugin_creator = trt.get_plugin_registry().get_creator(name, "1", "")
    if plugin_creator is None:
        print(f"Fail loading plugin {name}")
        return None
    field_list = []
    field_collection = trt.PluginFieldCollection(field_list)
    return plugin_creator.create_plugin(name, field_collection, trt.TensorRTPhase.BUILD)

def run():
    tw = TRTWrapperV1(trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:  # Create engine from scratch

        input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
        tw.profile.set_shape(input_tensor.name, [1, 1, 1], shape, shape)
        tw.config.add_optimization_profile(tw.profile)

        layer = tw.network.add_plugin_v3([input_tensor], [], getIdentityPlugin())
        tensor = layer.get_output(0)
        tensor.name = "outputT0"

        tw.build([tensor])
        tw.serialize_engine(trt_file)

    tw.setup(input_data)
    tw.infer(b_print_io=False)

    output_cpu = identity_cpu(input_data)

    check_array(tw.buffer["outputT0"][0], output_cpu["outputT0"], True)

if __name__ == "__main__":
    os.system("rm -rf *.trt")

    run()  # Build engine and plugin to do inference
    run()  # Load engine and plugin to do inference

    print("Finish")
