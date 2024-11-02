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
from pathlib import Path

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1, check_array

scalar = 1.0
shape = [3, 4, 5]
input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
trt_file = Path("model.trt")
timing_cache_file = Path("model.TimingCache")
plugin_file_list = [Path(__file__).parent / "AddScalarPlugin.so"]

def add_scalar_cpu(buffer, scalar):
    return {"outputT0": buffer["inputT0"] + scalar * 2}

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
    logger = trt.Logger(trt.Logger.Severity.VERBOSE)  # USe Verbose log to see more detail
    tw = TRTWrapperV1(logger=logger, trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:  # Create engine from scratch

        timing_cache_bytes = b""
        if timing_cache_file.exists():
            with open(timing_cache_file, "rb") as f:
                timing_cache_bytes = f.read()
            print(f"Succeed loading timing cache file {timing_cache_file}")
        else:
            print(f"Fail loading timing cache file {timing_cache_file}, we will create one during this building")
        timing_cache = tw.config.create_timing_cache(timing_cache_bytes)
        tw.config.set_timing_cache(timing_cache, False)

        input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
        tw.profile.set_shape(input_tensor.name, [1, 1, 1], shape, shape)
        tw.config.add_optimization_profile(tw.profile)

        # Use two same layer so you can see TensorRT only profile it once
        layer = tw.network.add_plugin_v3([input_tensor], [], getAddScalarPlugin(scalar))
        tensor = layer.get_output(0)

        layer = tw.network.add_plugin_v3([tensor], [], getAddScalarPlugin(scalar))
        tensor = layer.get_output(0)

        tensor.name = "outputT0"

        tw.build([tensor])
        tw.serialize_engine(trt_file)

        timingCache = tw.config.get_timing_cache()
        #print("timingCache.combine:%s" % res)

        timing_cache_bytes = timingCache.serialize()
        with open(timing_cache_file, "wb") as f:
            f.write(timing_cache_bytes)
            print(f"Succeed saving timing cache file {timing_cache_file}")

    print("#--------------------------------------------------------------------")

    tw.setup(input_data)
    tw.infer(b_print_io=True)

    output_cpu = add_scalar_cpu(input_data, scalar)

    check_array(tw.buffer["outputT0"][0], output_cpu["outputT0"], True)

if __name__ == "__main__":
    os.system("rm -rf *.trt *.TimingCache")
    run()  # Build engine and plugin to do inference
    run()  # Load engine and plugin to do inference

    # Remove engine and rebuild it with timing cache
    os.system("rm -rf *.trt")
    run()
    run()

    print("Finish")
