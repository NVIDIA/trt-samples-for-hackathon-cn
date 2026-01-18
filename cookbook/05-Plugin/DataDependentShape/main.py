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
from tensorrt_cookbook import TRTWrapperDDS, check_array, get_plugin, case_mark

@case_mark
def case_dds():
    shape = [4, 4]
    data = np.zeros(shape).astype(np.float32)
    data[0, 1] = 1
    data[0, 2] = 2
    data[1, 0] = 3
    data[1, 1] = 4
    data[1, 2] = 5
    data[2, 1] = 6
    data[2, 3] = 7
    input_data = {"inputT0": data}
    trt_file = Path("model.trt")
    plugin_file_list = [Path(__file__).parent / "PushLeftPlugin.so"]

    def push_left_cpu(buffer, epsilon=1.0e-5):
        input_buffer = buffer["inputT0"]
        batch_size, max_seq_len = input_buffer.shape
        max_out_seq_len = np.max(np.sum(np.abs(buffer["inputT0"]) > epsilon, axis=1))
        output_buffer = np.zeros([batch_size, max_out_seq_len], dtype=input_buffer.dtype)
        for i in range(batch_size):
            dst = 0
            for j in range(max_seq_len):
                if np.abs(input_buffer[i, j]) > epsilon:
                    output_buffer[i, dst] = input_buffer[i, j]
                    dst += 1
        return {"outputT0": output_buffer, "outputT1": max_out_seq_len}

    tw = TRTWrapperDDS(trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:  # Create engine from scratch

        plugin_info_dict = {
            "PushLeftPluginLayer": {
                "name": "PushLeft",
                "version": "1",
                "namespace": "",
                "argument_dict": {},
                "number_input_tensor": 1,
                "number_input_shape_tensor": 0,
            },
        }

        input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1])
        tw.profile.set_shape(input_tensor.name, [1, 1], shape, shape)
        tw.config.add_optimization_profile(tw.profile)

        layer = tw.network.add_plugin_v3([input_tensor], [], get_plugin(plugin_info_dict["PushLeftPluginLayer"]))
        layer.name = "PushLeftPluginLayer"
        tensor0 = layer.get_output(0)
        tensor0.name = "outputT0"
        tensor1 = layer.get_output(1)
        tensor1.name = "outputT1"

        tw.build([tensor0, tensor1])
        tw.serialize_engine(trt_file)

    tw.setup(input_data)
    tw.infer()

    output_cpu = push_left_cpu(input_data)

    for name in ["outputT0", "outputT1"]:
        check_array(tw.buffer[name][0], output_cpu[name], True, name)

if __name__ == "__main__":
    os.system("rm -rf *.trt")

    case_dds()
    case_dds()

    print("Finish")
