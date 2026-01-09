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
from tensorrt_cookbook import TRTWrapperV2, case_mark, check_array

shape = [3, 4, 5]
output_shape = [3, 2, 10]
input_data = {"inputT0": np.ones(shape, dtype=np.float32), "inputT1": np.array(output_shape, dtype=np.int32)}
onnx_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model" / "model-reshape.onnx"
trt_file = Path("model.trt")
plugin_file_list = [Path(__file__).parent / "MyReshapePlugin.so"]

def reshape_cpu(buffer, epsilon=1.0e-5):
    input_buffer = buffer["inputT0"]
    output_shape = buffer["inputT1"]
    return {"outputT0": input_buffer.reshape(output_shape)}

def get_my_reshape_plugin():
    name = "MyReshape"
    plugin_creator = trt.get_plugin_registry().get_creator(name, "1", "")
    if plugin_creator is None:
        print(f"Fail loading plugin {name}")
        return None
    field_list = []
    field_collection = trt.PluginFieldCollection(field_list)
    return plugin_creator.create_plugin(name, field_collection, trt.TensorRTPhase.BUILD)

@case_mark
def case_trt():
    tw = TRTWrapperV2(trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:  # Create engine from scratch

        tensor0 = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
        tensor1 = tw.network.add_input("inputT1", trt.int32, [len(output_shape)])
        tw.profile.set_shape(tensor0.name, [1, 1, 1], shape, shape)
        tw.profile.set_shape_input(tensor1.name, [1, 1, 1], output_shape, output_shape)  # range of value rather than shape
        tw.config.add_optimization_profile(tw.profile)

        layer = tw.network.add_plugin_v3([tensor0], [tensor1], get_my_reshape_plugin())
        tensor = layer.get_output(0)
        tensor.name = "outputT0"

        tw.build([tensor])
        tw.serialize_engine(trt_file)

    tw.setup(input_data)
    tw.infer()

    output_cpu = reshape_cpu(input_data)

    for name in ["outputT0"]:
        check_array(tw.buffer[name][0], output_cpu[name], True, name)

@case_mark
def case_onnx():
    tw = TRTWrapperV2(logger_level=trt.Logger.Severity.VERBOSE, trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:  # Create engine from scratch

        parser = trt.OnnxParser(tw.network, tw.logger)
        with open(onnx_file, "rb") as model:
            parser.parse(model.read())

        tensor0 = tw.network.get_input(0)
        tensor1 = tw.network.get_input(1)
        tw.profile.set_shape(tensor0.name, [1, 1, 1], shape, shape)
        tw.profile.set_shape_input(tensor1.name, [1, 1, 1], output_shape, output_shape)  # range of value rather than shape
        tw.config.add_optimization_profile(tw.profile)

        tw.build()
        tw.serialize_engine(trt_file)

    #tw.setup(input_data)
    #tw.infer()

    #output_cpu = reshape_cpu(input_data)

    #for name in ["outputT0"]:
    #    check_array(tw.buffer[name][0], output_cpu[name], True, name)

if __name__ == "__main__":
    os.system("rm -rf *.trt")
    case_trt()  # Build engine from TRT and plugin to do inference
    case_trt()  # Load engine and plugin to do inference
    #case_onnx()  # Build engine from ONNX and plugin to do inference, BUG in TensorRT
    #case_onnx()  # Load engine and plugin to do inference

    print("Finish")
