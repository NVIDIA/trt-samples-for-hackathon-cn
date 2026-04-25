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
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import TRTWrapperV1, case_mark, check_array, get_plugin

@case_mark
def case_simple():
    scalar = 1.0
    shape = [3, 4, 5]
    input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    trt_file = Path("model.trt")
    plugin_file_list = [Path(__file__).parent / "AddScalarPlugin.so"]

    def add_scalar_cpu(buffer, scalar):
        return {"outputT0": buffer["inputT0"] + scalar}

    tw = TRTWrapperV1(trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:  # Create tw.engine from scratch
        tw.config.set_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03, True)  # Use this switch to enable in-place plugin

        plugin_info_dict = {
            "AddScalarPluginLayer": dict(
                name="AddScalar",
                version="1",
                namespace="",
                argument_dict=dict(scalar=np.array([1.0], dtype=np.float32)),
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

    # Set output buffer to the same address as input
    tw.context.set_tensor_address("outputT0", tw.buffer["inputT0"][1])

    tw.infer()  # Notice the printed output is all zeros

    output_cpu = add_scalar_cpu(input_data, scalar)

    # Compare the first time, the result in CPU output buffer should be incorrect
    check_array(tw.buffer["outputT0"][0], output_cpu["outputT0"], True)

    # Copy GPU input buffer to CPU output buffer
    cudart.cudaMemcpyAsync(tw.buffer["outputT0"][0].ctypes.data, tw.buffer["inputT0"][1], tw.buffer["inputT0"][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, tw.stream)
    cudart.cudaStreamSynchronize(tw.stream)

    # Compare again, the result in CPU output buffer should be correct
    check_array(tw.buffer["outputT0"][0], output_cpu["outputT0"], True)

if __name__ == "__main__":
    for trt_path in Path(".").glob("*.trt"):
        trt_path.unlink(missing_ok=True)

    case_simple()
    case_simple()

    print("Finish")
