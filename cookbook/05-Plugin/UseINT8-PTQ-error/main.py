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

import os
from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import MyCalibratorV1, TRTWrapperV1, check_array

scalar = 1.0
shape = [1, 8, 2, 3]  # CHW4 format needs input tensor with at least 4 Dimensions
input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
trt_file = Path("model.trt")
int8_cache_file = Path("model.Int8Cache")
plugin_file_list = [Path(__file__).parent / "AddScalarPlugin.so"]

def add_scalar_cpu(buffer, scalar):
    return {"outputT0": buffer["inputT0"] + scalar}

def getAddScalarPlugin(scalar):
    name = "AddScalar"
    plugin_creator = trt.get_plugin_registry().get_creator(name, "1", "")
    if plugin_creator is None:
        print(f"Fail loading plugin {name}")
        return None
    field_list = []
    field_list.append(trt.PluginField("scalar", np.array([scalar], dtype=np.float32), trt.PluginFieldType.FLOAT32))
    field_collection = trt.PluginFieldCollection(field_list)
    return plugin_creator.create_plugin(name, field_collection, trt.TensorRTPhase.BUILD)

def run():
    tw = TRTWrapperV1(trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:  # Create engine from scratch
        tw.config.set_flag(trt.BuilderFlag.INT8)
        tw.config.int8_calibrator = MyCalibratorV1(1, shape, int8_cache_file)

        input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])
        tw.profile.set_shape(input_tensor.name, [1, 1, 1, 1], shape, shape)
        tw.config.add_optimization_profile(tw.profile)
        #md = np.max(input_data["inputT0"]) + 1
        #input_tensor.dynamic_range = [-md, md]  # set dynamic range if calibrator is not used

        layer = tw.network.add_plugin_v3([input_tensor], [], getAddScalarPlugin(scalar))
        tensor = layer.get_output(0)
        #tensor.dynamic_range = [-md - 5, md + 5]  # set dynamic range if calibrator is not used

        # Insert a cast layer to get float32 output, or int8 directly from plugin layer
        layer = tw.network.add_cast(tensor, trt.float32)
        tensor = layer.get_output(0)
        tensor.name = "outputT0"

        tw.build([tensor])
        tw.serialize_engine(trt_file)

    tw.setup(input_data)
    tw.infer(b_print_io=False)

    output_cpu = add_scalar_cpu(input_data, scalar)

    check_array(tw.buffer["outputT0"][0], output_cpu["outputT0"], True)

if __name__ == "__main__":
    os.system("rm -rf *.Int8Cache *.trt")

    run()  # Build engine and plugin to do inference
    run()  # Load engine and plugin to do inference

    print("Finish")
