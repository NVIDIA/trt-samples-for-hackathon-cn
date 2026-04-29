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
def case_int8_qdq_plugin_skeleton():
    scalar = 1.0
    shape = [1, 8]
    input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    trt_file = Path("model.trt")
    plugin_file_list = [Path(__file__).parent.parent / "UseINT8-PTQ" / "AddScalarPlugin.so"]

    def add_scalar_cpu(buffer, value):
        return {"outputT0": buffer["inputT0"] + value}

    if not plugin_file_list[0].is_file():
        raise FileNotFoundError(f"Plugin .so not found: {plugin_file_list[0]}. Run `make -C ../UseINT8-PTQ build` first.")

    tw = TRTWrapperV1(trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:
        tw.config.set_flag(trt.BuilderFlag.INT8)
        input_tensor = tw.network.add_input("inputT0", trt.float32, shape)

        scale = tw.network.add_constant([1], np.array([0.1], dtype=np.float32)).get_output(0)
        q = tw.network.add_quantize(input_tensor, scale)
        dq = tw.network.add_dequantize(q.get_output(0), scale)

        plugin_info = {
            "name": "AddScalar",
            "version": "1",
            "namespace": "",
            "argument_dict": {
                "scalar": np.array([scalar], dtype=np.float32)
            },
            "number_input_tensor": 1,
            "number_input_shape_tensor": 0,
            "plugin_api_version": "3",
        }
        plugin = get_plugin(plugin_info)
        if plugin is None:
            raise RuntimeError("Plugin creator not found: AddScalar")
        layer = tw.network.add_plugin_v3([dq.get_output(0)], [], plugin)
        out = layer.get_output(0)
        out.name = "outputT0"

        tw.build([out])
        tw.serialize_engine(trt_file)
    tw.setup(input_data)
    tw.infer()

    output_cpu = add_scalar_cpu(input_data, scalar)
    check_array(tw.buffer["outputT0"][0], output_cpu["outputT0"], True)

if __name__ == "__main__":
    for trt_path in Path(".").glob("*.trt"):
        trt_path.unlink(missing_ok=True)
    case_int8_qdq_plugin_skeleton()
    case_int8_qdq_plugin_skeleton()
    print("Finish")
