#
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
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1, check_array

scalar = 1.0
shape = [3, 4, 5]
input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
trt_file = Path("model.trt")
plugin_file_list = [Path(__file__).parent / "PassHostDataPlugin.so"]

def add_scalar_cpu(buffer, scalar):
    return {"outputT0": buffer["inputT0"] + scalar}

def getPassHostDataPlugin(scalar):
    name = "PassHostData"
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

        input_tensor = tw.network.add_input("inputT0", trt.int64, [1])  # add an input tensor of int64 type

        layer = tw.network.add_plugin_v3([input_tensor], [], getPassHostDataPlugin(scalar))
        tensor = layer.get_output(0)
        tensor.name = "outputT0"

        tw.build([tensor])
        tw.serialize_engine(trt_file)

    # We `tw.setup` and `tw.`
    tw.setup(input_data)
    tw.infer(b_print_io=False)

    output_cpu = add_scalar_cpu(input_data, scalar)

    check_array(tw.buffer["outputT0"][0], output_cpu["outputT0"], True)

if __name__ == "__main__":
    os.system("rm -rf *.trt")

    run()  # Build engine and plugin to do inference
    run()  # Load engine and plugin to do inference

    print("Finish")
"""
        for name in self.tensor_name_list:
            if name == "inputT1":
                data = input_data["inputT1"]
                print(f"[python]{data=}")
                p_data = data.ctypes.data
                print(f"[python]{p_data=}")
                buffer = np.array([p_data], dtype=np.int64)
                p_p_data = buffer.ctypes.data
                print(f"[python]{p_p_data=}")
                self.context.set_tensor_address(name, 0)
            else:
                self.context.set_tensor_address(name, self.buffer[name][1])

        return

    def infer(self, b_print_io: bool = True, b_get_timeline: bool = False) -> None:
        # Do inference and print output
        for name in self.tensor_name_list:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name == "inputT1":
                    continue
                else:
                    cudart.cudaMemcpy(self.buffer[name][1], self.buffer[name][0].ctypes.data, self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
"""
