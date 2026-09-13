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

from tensorrt_cookbook import TRTWrapperV1, case_mark, check_array

so_file = Path(__file__).parent / "LayerNormPluginOneFlow.so"
plugin_version = "5"
epsilon = 1e-6
np.random.seed(31193)

def layer_norm_cpu(buffer, epsilon):
    x = buffer[0]
    mean = np.mean(x, 2)[:, :, np.newaxis]
    diff = x - mean
    var = np.mean(diff * diff, 2)[:, :, np.newaxis]
    output = diff / np.sqrt(var + epsilon)
    return [output]

def get_layer_norm_plugin(epsilon):
    for creator in trt.get_plugin_registry().plugin_creator_list:
        if creator.name == "LayerNorm" and creator.plugin_version == plugin_version:
            print(f"Find {creator.name} V{creator.plugin_version}")
            field_list = [trt.PluginField("epsilon", np.float32(epsilon), trt.PluginFieldType.FLOAT32)]
            return creator.create_plugin(creator.name, trt.PluginFieldCollection(field_list))
    return None

@case_mark
def case_simple(shape, b_fp16):
    trt_file = Path(f"model-{shape[2]}-{'FP16' if b_fp16 else 'FP32'}.plan")
    trt_dtype = trt.float16 if b_fp16 else trt.float32
    np_dtype = np.float16 if b_fp16 else np.float32
    input_data = {"inputT0": np.random.rand(np.prod(shape)).astype(np_dtype).reshape(shape)}

    tw = TRTWrapperV1(trt_file=trt_file, plugin_file_list=[so_file])
    if tw.engine_bytes is None:  # Create engine from scratch
        input_tensor = tw.network.add_input("inputT0", trt_dtype, [-1 for _ in shape])
        tw.profile.set_shape(input_tensor.name, [1, 1, shape[2]], shape, shape)

        plugin_layer = tw.network.add_plugin_v2([input_tensor], get_layer_norm_plugin(epsilon))
        tensor = plugin_layer.get_output(0)
        tensor.name = "outputT0"

        tw.build([tensor])
        tw.serialize_engine(trt_file)

    tw.setup(input_data)
    tw.infer()

    output_cpu = layer_norm_cpu([input_data["inputT0"]], epsilon)
    check_array(tw.buffer["outputT0"][0], output_cpu[0], True)

if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=200, suppress=True)
    for plan_path in Path(".").glob("*.plan"):
        plan_path.unlink(missing_ok=True)

    case_simple([1, 1, 256], False)
    case_simple([16, 64, 256], False)
    case_simple([1, 1, 256], True)
    case_simple([16, 64, 256], True)

    print("Finish")
