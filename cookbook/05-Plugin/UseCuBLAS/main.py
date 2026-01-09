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
from tensorrt_cookbook import TRTWrapperV1, check_array

b, m, k, n = 5, 2, 3, 4
input_data = {"inputT0": np.random.rand(b * m * k).astype(np.float32).reshape(b, m, k) * 2 - 1}
weight = np.random.rand(k * n).astype(np.float32).reshape(k, n) * 2 - 1
trt_file = Path("model.trt")
plugin_file_list = [Path(__file__).parent / "CuBLASGemmPlugin.so"]

def cuBLAS_gemm_cpu(buffer, weight):
    return {"outputT0": np.matmul(buffer["inputT0"], weight)}

def getCuBLASGemmPlugin(weight):
    name = "CuBLASGemm"
    plugin_creator = trt.get_plugin_registry().get_creator(name, "1", "")
    if plugin_creator is None:
        print(f"Fail loading plugin {name}")
        return None
    parameterList = []
    parameterList.append(trt.PluginField("K", np.int32(weight.shape[0]), trt.PluginFieldType.INT32))
    parameterList.append(trt.PluginField("N", np.int32(weight.shape[1]), trt.PluginFieldType.INT32))
    parameterList.append(trt.PluginField("Weight", np.float32(weight), trt.PluginFieldType.FLOAT32))
    field_collection = trt.PluginFieldCollection(parameterList)
    return plugin_creator.create_plugin(name, field_collection, trt.TensorRTPhase.BUILD)

def run():
    tw = TRTWrapperV1(trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:  # Create engine from scratch

        input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, k])
        tw.profile.set_shape(input_tensor.name, [1, 1, k], [b, m, k], [b * 2, m * 2, k])
        tw.config.add_optimization_profile(tw.profile)

        layer = tw.network.add_plugin_v3([input_tensor], [], getCuBLASGemmPlugin(weight))
        tensor = layer.get_output(0)
        tensor.name = "outputT0"

        tw.build([tensor])
        tw.serialize_engine(trt_file)

    tw.setup(input_data)
    tw.infer(b_print_io=False)

    output_cpu = cuBLAS_gemm_cpu(input_data, weight)

    check_array(tw.buffer["outputT0"][0], output_cpu["outputT0"], True)

if __name__ == "__main__":
    os.system("rm -rf ./*.trt")

    run()  # build TensorRT engine and do inference
    run()  # load TensorRT engine and do inference

    print("Test all finish!")
