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

# Migrate a Python plugin from the deprecated `IPluginV2DynamicExt` to `IPluginV3`.
#
# The same operator (Y = scale * X) is implemented twice, once per interface, sharing one CUDA
# kernel. Reading the two classes side by side *is* the migration guide; see README.md for the
# method-by-method mapping table. `IPluginV2DynamicExt` has been deprecated since TensorRT-8.5 and
# is scheduled for removal in TensorRT-12, so existing plugins should move to `IPluginV3`.

import ctypes
import json
from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt
from cuda.bindings import driver as cuda
from tensorrt_cookbook import TRTWrapperV1, case_mark, ceil_divide, check_array, check_nvrtc_error, get_kernel

# The two plugins register under distinct names so both can live in the plugin registry at once.
PLUGIN_NAME_V2 = "ScaleV2"
PLUGIN_NAME_V3 = "ScaleV3"
PLUGIN_VERSION = "1"

scale = 2.0
shape = [3, 4, 5]
input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}

def scale_cpu(buffer, scale):
    return {"outputT0": buffer["inputT0"] * scale}

# One kernel, shared by both plugin versions - the migration is purely about the interface.
source_code = r"""
extern "C" __global__
void scaleKernel(float const *x, float *y, float const scale, int const nElement)
{
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nElement)
        return;
    y[index] = x[index] * scale;
}
"""

def launch_scale_kernel(device, input_desc, inputs, outputs, scale, stream):
    """Launch the shared kernel. Identical in both plugins, so `enqueue()` is *not* what changes."""
    n_element = int(np.prod(np.array(input_desc[0].dims)))
    kernel = get_kernel(source_code, device, b"scaleKernel")
    block_size = 256
    grid_size = ceil_divide(n_element, block_size)

    arg_values = (inputs[0], outputs[0], np.float32(scale), np.int32(n_element))
    arg_types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int32)
    kernel_args = (arg_values, arg_types)

    check_nvrtc_error(cuda.cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, cuda.CUstream(stream), kernel_args, 0))

# ==================================================================================================
# Before: the deprecated `IPluginV2DynamicExt` interface
# ==================================================================================================

class ScalePluginV2(trt.IPluginV2DynamicExt):

    def __init__(self, scale: float):
        trt.IPluginV2DynamicExt.__init__(self)
        self.plugin_type = PLUGIN_NAME_V2  # V2 spells the name `plugin_type`
        self.plugin_version = PLUGIN_VERSION
        self.plugin_namespace = ""
        self.num_outputs = 1
        self.scale = scale
        self.device = 0

    # V2 owns its serialization: it must report the byte count and produce the bytes itself.
    def get_serialization_size(self) -> int:
        return len(json.dumps({"scale": self.scale}))

    def serialize(self) -> bytes:
        return json.dumps({"scale": self.scale})

    # V2 has an explicit resource lifecycle.
    def initialize(self) -> int:
        return 0

    def terminate(self) -> None:
        return

    def destroy(self) -> None:
        return

    # V2 reports one output datatype at a time.
    def get_output_datatype(self, output_index: int, input_types: List[trt.DataType]) -> trt.DataType:
        return input_types[0]

    # V2 reports one output shape at a time, returning a bare `DimsExprs`.
    def get_output_dimensions(self, output_index: int, inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> trt.DimsExprs:
        return trt.DimsExprs(inputs[0])  # Elementwise, so the output shape equals the input shape

    def configure_plugin(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> None:
        return

    # V2 gives `PluginTensorDesc` elements, so `.type` / `.format` are read directly.
    def supports_format_combination(self, pos: int, in_out: List[trt.PluginTensorDesc], num_inputs: int) -> bool:
        desc = in_out[pos]
        if desc.format != trt.TensorFormat.LINEAR:
            return False
        if pos == 0:
            return desc.type == trt.float32
        return desc.type == in_out[0].type

    def get_workspace_size(self, ptd_in: List[trt.PluginTensorDesc], ptd_out: List[trt.PluginTensorDesc]) -> int:
        return 0

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc], inputs: List[int], outputs: List[int], workspace: int, stream: int) -> None:
        launch_scale_kernel(self.device, input_desc, inputs, outputs, self.scale, stream)

    def clone(self) -> trt.IPluginV2DynamicExt:
        cloned = ScalePluginV2(self.scale)
        cloned.__dict__.update(self.__dict__)
        return cloned

class ScalePluginV2Creator(trt.IPluginCreator):

    def __init__(self):
        trt.IPluginCreator.__init__(self)
        self.name = PLUGIN_NAME_V2
        self.plugin_version = PLUGIN_VERSION
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection([trt.PluginField("scale", np.array([]), trt.PluginFieldType.FLOAT32)])

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection):
        scale = 1.0
        for field in field_collection:
            if field.name == "scale":
                scale = float(field.data[0])
        return ScalePluginV2(scale)

    # V2 creators must rebuild the plugin from the bytes `serialize()` wrote.
    def deserialize_plugin(self, name: str, data: bytes):
        deserialized = ScalePluginV2(1.0)
        deserialized.__dict__.update(dict(json.loads(data)))
        return deserialized

# ==================================================================================================
# After: the `IPluginV3` interface
# ==================================================================================================

# One V2 class becomes `IPluginV3` plus three capability interfaces. They may live in separate
# classes; here a single object implements all of them, so `get_capability_interface` returns self.
class ScalePluginV3(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):

    def __init__(self, scale: float):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)
        self.plugin_name = PLUGIN_NAME_V3  # V3 spells the name `plugin_name` (V2 used `plugin_type`)
        self.plugin_version = PLUGIN_VERSION
        self.plugin_namespace = ""
        self.num_outputs = 1
        self.scale = scale
        self.device = 0

    # IPluginV3: hand TensorRT the interface for the phase it is asking about.
    def get_capability_interface(self, plugin_capability_type: trt.PluginCapabilityType) -> trt.IPluginCapability:
        return self

    # IPluginV3OneBuild: all output datatypes at once, as a list.
    def get_output_data_types(self, input_types: List[trt.DataType]) -> List[trt.DataType]:
        return [input_types[0]]

    # IPluginV3OneBuild: all output shapes at once, as a list.
    def get_output_shapes(self, inputs: List[trt.DimsExprs], shape_inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> List[trt.DimsExprs]:
        return [trt.DimsExprs(inputs[0])]

    # IPluginV3OneBuild: there is no initialize()/terminate() pair any more - acquire per-shape
    # resources here and in `on_shape_change()` instead.
    def configure_plugin(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> None:
        return

    # IPluginV3OneBuild: elements are `DynamicPluginTensorDesc` now, so the static part of the
    # descriptor is one level deeper, behind `.desc`.
    def supports_format_combination(self, pos: int, in_out: List[trt.DynamicPluginTensorDesc], num_inputs: int) -> bool:
        desc = in_out[pos].desc
        if desc.format != trt.TensorFormat.LINEAR:
            return False
        if pos == 0:
            return desc.type == trt.float32
        return desc.type == in_out[0].desc.type

    def get_workspace_size(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> int:
        return 0

    def get_valid_tactics(self) -> List[int]:
        return [1]

    def set_tactic(self, tactic: int) -> None:
        return

    # IPluginV3OneRuntime: replaces the V2 attach_to_context / detach_from_context pair and returns
    # the per-context clone rather than mutating self.
    def attach_to_context(self, resource_context: trt.IPluginResourceContext) -> trt.IPluginV3:
        return self.clone()

    def on_shape_change(self, ptd_in: List[trt.PluginTensorDesc], ptd_out: List[trt.PluginTensorDesc]) -> None:
        return

    # IPluginV3OneRuntime: the biggest win of the migration. Declare the fields and TensorRT does
    # the serializing *and* re-creates the plugin through the creator - no `serialize()` on the
    # plugin, no `deserialize_plugin()` on the creator, no manual size accounting.
    def get_fields_to_serialize(self) -> trt.PluginFieldCollection:
        return trt.PluginFieldCollection([trt.PluginField("scale", np.array(self.scale, dtype=np.float32), trt.PluginFieldType.FLOAT32)])

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc], inputs: List[int], outputs: List[int], workspace: int, stream: int) -> None:
        launch_scale_kernel(self.device, input_desc, inputs, outputs, self.scale, stream)

    def clone(self) -> trt.IPluginV3:
        cloned = ScalePluginV3(self.scale)
        cloned.__dict__.update(self.__dict__)
        return cloned

class ScalePluginV3Creator(trt.IPluginCreatorV3One):  # V2 used `trt.IPluginCreator`

    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = PLUGIN_NAME_V3
        self.plugin_version = PLUGIN_VERSION
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection([trt.PluginField("scale", np.array([]), trt.PluginFieldType.FLOAT32)])

    # V3 adds the build/runtime `phase` argument to `create_plugin`.
    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection, phase: trt.TensorRTPhase):
        scale = 1.0
        for field in field_collection:
            if field.name == "scale":
                scale = float(field.data[0])
        return ScalePluginV3(scale)

# ==================================================================================================

def build_and_run(plugin_name, trt_file):
    """Build a one-plugin engine for `plugin_name`, run it, and return the output."""
    tw = TRTWrapperV1(trt_file=trt_file)
    if tw.engine_bytes is None:
        plugin_creator = trt.get_plugin_registry().get_creator(plugin_name, PLUGIN_VERSION, "")
        field_collection = trt.PluginFieldCollection([trt.PluginField("scale", np.array(scale, dtype=np.float32), trt.PluginFieldType.FLOAT32)])

        input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
        tw.profile.set_shape(input_tensor.name, [1, 1, 1], shape, shape)

        if plugin_name == PLUGIN_NAME_V2:
            # V2: creator takes (name, field_collection); the layer takes input tensors only.
            plugin = plugin_creator.create_plugin(plugin_name, field_collection)
            layer = tw.network.add_plugin_v2([input_tensor], plugin)
        else:
            # V3: creator takes an extra `phase`; the layer takes input *and* shape-input tensors.
            plugin = plugin_creator.create_plugin(plugin_name, field_collection, trt.TensorRTPhase.BUILD)
            layer = tw.network.add_plugin_v3([input_tensor], [], plugin)

        tensor = layer.get_output(0)
        tensor.name = "outputT0"
        tw.build([tensor])
        tw.serialize_engine(trt_file)

    tw.setup(input_data)
    tw.infer(b_print_io=False)
    return tw.buffer["outputT0"][0]

output_dict = {}  # Keep the two outputs out of the case arguments, which `case_mark` echoes

@case_mark
def case_v2():
    """Build and run the plugin written against the deprecated `IPluginV2DynamicExt`."""
    trt_file = Path("model-v2.trt")
    output_dict["v2"] = build_and_run(PLUGIN_NAME_V2, trt_file)
    check_array(output_dict["v2"], scale_cpu(input_data, scale)["outputT0"], True)

    # Re-run from the serialized engine. This is the path that exercises the V2 serialization
    # contract: TensorRT hands the bytes `ScalePluginV2.serialize()` wrote back to
    # `ScalePluginV2Creator.deserialize_plugin()`, which must reconstruct the plugin by itself.
    print("Reload from the serialized engine (goes through deserialize_plugin):")
    check_array(build_and_run(PLUGIN_NAME_V2, trt_file), output_dict["v2"], True)

@case_mark
def case_v3():
    """Build and run the same operator written against `IPluginV3`."""
    trt_file = Path("model-v3.trt")
    output_dict["v3"] = build_and_run(PLUGIN_NAME_V3, trt_file)
    check_array(output_dict["v3"], scale_cpu(input_data, scale)["outputT0"], True)

    # Same reload, but there is no deserialization code in this file at all: TensorRT stored the
    # fields from `get_fields_to_serialize()` and rebuilds the plugin through the creator's
    # `create_plugin(..., phase=trt.TensorRTPhase.RUNTIME)`.
    print("Reload from the serialized engine (goes through the creator, no deserialize_plugin):")
    check_array(build_and_run(PLUGIN_NAME_V3, trt_file), output_dict["v3"], True)

@case_mark
def case_compare():
    """The point of the migration: V3 must reproduce V2 bit for bit."""
    check_array(output_dict["v3"], output_dict["v2"], True)

if __name__ == "__main__":
    for trt_path in Path(".").glob("*.trt"):
        trt_path.unlink(missing_ok=True)

    # Register both creators once, so the two plugins coexist in the registry.
    plugin_registry = trt.get_plugin_registry()
    registered_name_list = [creator.name for creator in plugin_registry.all_creators]
    for creator in [ScalePluginV2Creator(), ScalePluginV3Creator()]:
        if creator.name not in registered_name_list:
            plugin_registry.register_creator(creator, "")

    case_v2()
    case_v3()
    case_compare()

    print("Finish")
