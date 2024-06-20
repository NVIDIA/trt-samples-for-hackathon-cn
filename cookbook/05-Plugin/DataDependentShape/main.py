#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
#

import os
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperDDS, check_array

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

def getPushLeftPlugin():
    name = "PushLeft"
    plugin_creator = trt.get_plugin_registry().get_creator(name, "1", "")
    if plugin_creator is None:
        print(f"Fail loading plugin {name}")
        return None
    field_list = []
    field_collection = trt.PluginFieldCollection(field_list)
    return plugin_creator.create_plugin(name, field_collection, trt.TensorRTPhase.BUILD)

def run():
    tw = TRTWrapperDDS(plugin_file_list=plugin_file_list, trt_file=trt_file)
    if tw.engine_bytes is None:  # Create engine from scratch

        input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1])
        tw.profile.set_shape(input_tensor.name, [1, 1], shape, shape)
        tw.config.add_optimization_profile(tw.profile)

        layer = tw.network.add_plugin_v3([input_tensor], [], getPushLeftPlugin())
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

    run()  # Build engine and plugin to do inference
    run()  # Load engine and plugin to do inference

    print("Finish")
