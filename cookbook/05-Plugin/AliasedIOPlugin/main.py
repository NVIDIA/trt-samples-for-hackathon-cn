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

# A plugin that writes into one of its own inputs, using aliased I/O.
#
# Before `IPluginV3OneBuildV2`, plugin inputs were strictly read-only, which put in-place operators
# out of reach: the plugin had to copy the whole input to the output first, even when it only
# touched a handful of elements. `IPluginV3OneBuildV2` adds `get_aliased_input(output_index)`, which
# declares that an output shares its buffer with an input, so the plugin can update it in place.
#
# The operator here is scatter-add - `data[index[i]] += updates[i]` - which is the aggregation step
# of a graph neural network: each node sums the features of its neighbours. `data` is the
# accumulator, so being able to add into it directly (instead of copying it first) is the whole
# point.
#
# Related examples: `05-Plugin/InPlacePlugin` is the same capability written in C++
# (`v_2_0::IPluginV3OneBuild` + `getAliasedInput`), but its operator is an elementwise AddScalar
# that would work just as well without aliasing; scatter-add is an op that genuinely needs to
# accumulate into its own input. `02-API/Layer/KVCacheUpdate` and `02-API/CudaEngine` cover the
# engine side (`get_aliased_input_tensor`).

import ctypes
from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import TRTWrapperV1, case_mark, ceil_divide, check_array, check_nvrtc_error, get_kernel

trt_file = Path("model.trt")

data = {
    "data": np.zeros(6, dtype=np.float32),  # Accumulator, aliased with the output
    "index": np.array([1, 2, 3, 0, 2, 3], dtype=np.int32),  # Destination node of each neighbour
    "updates": np.array([1.0, 3.0, 5.0, 7.0, 1.0, 3.0], dtype=np.float32),  # Neighbour features
}

def scatter_add_cpu(buffer):
    output = buffer["data"].copy()
    np.add.at(output, buffer["index"], buffer["updates"])  # `+=` would drop duplicate indices
    return {"data_out": output}

# `atomicAdd` because several neighbours may target the same node (index 2 and 3 appear twice here).
source_code = r"""
extern "C" __global__
void scatterAddKernel(float *data, int const *index, float const *updates, int const nUpdate)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nUpdate)
        return;
    atomicAdd(data + index[i], updates[i]);
}
"""

# `IPluginV3OneBuildV2` replaces `IPluginV3OneBuild`; everything else is the usual V3 plugin.
class ScatterAddPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuildV2, trt.IPluginV3OneRuntime):

    def __init__(self):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuildV2.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)
        self.plugin_name = "ScatterAdd"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.num_outputs = 1
        self.device = 0

    def get_capability_interface(self, plugin_capability_type: trt.PluginCapabilityType) -> trt.IPluginCapability:
        return self

    # The one method that `IPluginV3OneBuildV2` adds: output 0 shares its buffer with input 0.
    # Return -1 (or any negative value) for an output that is not aliased.
    def get_aliased_input(self, output_index: int) -> int:
        return 0 if output_index == 0 else -1

    def get_output_data_types(self, input_types: List[trt.DataType]) -> List[trt.DataType]:
        return [input_types[0]]

    def get_output_shapes(self, inputs: List[trt.DimsExprs], shape_inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> List[trt.DimsExprs]:
        return [trt.DimsExprs(inputs[0])]  # An aliased output necessarily has the shape of the input it aliases

    def supports_format_combination(self, pos: int, in_out: List[trt.DynamicPluginTensorDesc], num_inputs: int) -> bool:
        desc = in_out[pos].desc
        if desc.format != trt.TensorFormat.LINEAR:
            return False
        if pos == 1:  # `index`
            return desc.type == trt.int32
        return desc.type == trt.float32  # `data`, `updates` and the aliased output

    def configure_plugin(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> None:
        return

    def get_workspace_size(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> int:
        return 0

    def get_valid_tactics(self) -> List[int]:
        return [1]

    def set_tactic(self, tactic: int) -> None:
        return

    def on_shape_change(self, ptd_in: List[trt.PluginTensorDesc], ptd_out: List[trt.PluginTensorDesc]) -> None:
        return

    def attach_to_context(self, resource_context: trt.IPluginResourceContext) -> trt.IPluginV3:
        return self.clone()

    def get_fields_to_serialize(self) -> trt.PluginFieldCollection:
        return trt.PluginFieldCollection([])

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc], inputs: List[int], outputs: List[int], workspace: int, stream: int) -> None:
        # TensorRT guarantees `outputs[0] == inputs[0]` because of `get_aliased_input`, so the
        # kernel updates the accumulator in place and never copies it.
        assert outputs[0] == inputs[0], f"Aliasing broken: {inputs[0]=} != {outputs[0]=}"

        n_update = int(np.prod(np.array(input_desc[2].dims)))
        kernel = get_kernel(source_code, self.device, b"scatterAddKernel")
        block_size = 256
        grid_size = ceil_divide(n_update, block_size)

        arg_values = (outputs[0], inputs[1], inputs[2], np.int32(n_update))
        arg_types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32)
        check_nvrtc_error(cuda.cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, cuda.CUstream(stream), (arg_values, arg_types), 0))

    def clone(self) -> trt.IPluginV3:
        cloned = ScatterAddPlugin()
        cloned.__dict__.update(self.__dict__)
        return cloned

class ScatterAddPluginCreator(trt.IPluginCreatorV3One):

    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = "ScatterAdd"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection([])

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection, phase: trt.TensorRTPhase):
        return ScatterAddPlugin()

def run_scatter_add(data):
    tw = TRTWrapperV1(trt_file=trt_file)
    if tw.engine_bytes is None:
        # Without this switch the build fails with error code 4:
        #   "Aliased I/O used by plugin ... but PreviewFeature::kALIASED_PLUGIN_IO_10_03 not enabled"
        tw.builder_config.set_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03, True)

        plugin_creator = trt.get_plugin_registry().get_creator("ScatterAdd", "1", "")
        plugin = plugin_creator.create_plugin("ScatterAdd", trt.PluginFieldCollection([]), trt.TensorRTPhase.BUILD)

        input_tensor_list = [
            tw.network.add_input("data", trt.float32, data["data"].shape),
            tw.network.add_input("index", trt.int32, data["index"].shape),
            tw.network.add_input("updates", trt.float32, data["updates"].shape),
        ]
        layer = tw.network.add_plugin_v3(input_tensor_list, [], plugin)
        tensor = layer.get_output(0)
        tensor.name = "data_out"

        tw.build([tensor])
        tw.serialize_engine(trt_file)

    tw.setup(data)

    # `ICudaEngine.get_aliased_input_tensor()` reports aliasing introduced by *layers* such as
    # KVCacheUpdate (see `02-API/Layer/KVCacheUpdate`). It does NOT report plugin aliasing, so it
    # returns None here even though the plugin declared `get_aliased_input(0) == 0`.
    print(f"engine.get_aliased_input_tensor('data_out') = {tw.engine.get_aliased_input_tensor('data_out')}")

    # The caller is therefore responsible for honouring the aliasing contract: point the output at
    # the device buffer of the input it aliases. Otherwise the plugin writes into `data`'s buffer
    # while TensorRT reports a separate, untouched output buffer.
    tw.context.set_tensor_address("data_out", tw.buffer["data"][1])

    tw.infer()

    # The result now lives in `data`'s device buffer, not in the output's host buffer, so read it
    # back from there. `tw.buffer[name]` is (host array, device pointer, byte count).
    output = np.empty_like(data["data"])
    cudart.cudaMemcpy(output.ctypes.data, tw.buffer["data"][1], tw.buffer["data"][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    check_array(output, scatter_add_cpu(data)["data_out"], True)

@case_mark
def case_simple():
    """Aggregate into a zeroed accumulator: the classic GNN neighbourhood sum."""
    run_scatter_add(data)

@case_mark
def case_accumulate():
    """A non-zero accumulator shows the plugin really *adds into* the input rather than overwriting
    it - which is exactly what aliased I/O buys and what a read-only input could not express."""
    biased_data = dict(data, data=np.array([10, 20, 30, 40, 50, 60], dtype=np.float32))
    run_scatter_add(biased_data)

if __name__ == "__main__":
    trt_file.unlink(missing_ok=True)

    plugin_registry = trt.get_plugin_registry()
    my_plugin_creator = ScatterAddPluginCreator()
    if my_plugin_creator.name not in [creator.name for creator in plugin_registry.all_creators]:
        plugin_registry.register_creator(my_plugin_creator, "")

    case_simple()
    case_accumulate()

    print("Finish")
