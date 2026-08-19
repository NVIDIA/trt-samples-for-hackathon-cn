# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
"""One Python plugin, two backends, and let the **builder** pick which one ships.

The other files in this directory each pin one library: `add_scalar_torch.py` always
runs Torch, `add_scalar_triton.py` always runs Triton. Choosing between them is then a
human decision made once, off-line, on a guess.

`IPluginV3OneBuild.get_valid_tactics` turns that into a measurement. Return more than one
tactic and TensorRT times each of them during `build_serialized_network`, exactly as it
does for its own kernels, and bakes the winner into the engine. At run time `set_tactic`
is called once with the tactic that won.

Two things this file shows that the single-backend examples cannot:

1. **Autotuning across Python libraries.** Tactic 1 is Torch, tactic 2 is Triton, the
   builder decides. Watch the log: `set_tactic` is called repeatedly while building
   (timing) and exactly once when the engine runs (the winner).
2. **Per-format tactics.** `get_valid_tactics` is called *after* `configure_plugin`, so it
   can look at the I/O type that is being considered and offer a different tactic list per
   format. Here float16 is restricted to Triton, which is how you express "my Torch path
   does not handle this dtype well" without giving up the float32 competition.

Measured behaviour on TensorRT 11.1.0.106:

    float32 -> get_valid_tactics returns [Torch, Triton] -> 13 set_tactic calls while building
    float16 -> get_valid_tactics returns [Triton]        ->  0 set_tactic calls while building

A one-element tactic list is not "time one candidate", it is "skip timing". That is why the
other files in this directory, which all return `[1]`, never pay a tuning cost.

`05-Plugin/Tactic+TimingCache` does the same thing from C++ and adds a timing cache.
"""

import ctypes
from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt
import torch
import triton
import triton.language as tl
from cuda.bindings import runtime as cudart

from tensorrt_cookbook import TRTWrapperV1, ceil_divide, check_array

scalar = 1.0
shape = [3, 4, 5]
input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}

TACTIC_TORCH = 1
TACTIC_TRITON = 2
TACTIC_NAME = {TACTIC_TORCH: "Torch", TACTIC_TRITON: "Triton"}

# `True` -> float16 may only use Triton, float32 may use both
PER_FORMAT_TACTIC = True

tactic_log = []  # (phase, tactic, dtype) recorded from `set_tactic`, for the report

def add_scalar_cpu(buffer, scalar):
    return {"outputT0": buffer["inputT0"] + scalar}

@triton.jit
def add_scalar_triton_kernel(X, Y, scalar, n_element, BLOCK_SIZE: tl.constexpr):
    """The Triton backend, identical to the one in `add_scalar_triton.py`."""
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i < n_element
    x = tl.load(X + i, mask=mask)
    tl.store(Y + i, x + scalar, mask=mask)

class AddScalarPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):

    def __init__(self, scalar: float):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)
        self.plugin_name = "AddScalarMultiTactic"
        self.plugin_version = "1"
        self.num_outputs = 1
        self.plugin_namespace = ""
        self.scalar = scalar
        self.tactic = TACTIC_TORCH  # Overwritten by `set_tactic`
        self.current_type = trt.float32  # Recorded by `configure_plugin`, read by `get_valid_tactics`
        return

    def get_capability_interface(self, plugin_capability_type: trt.PluginCapabilityType) -> trt.IPluginCapability:
        return self

    def clone(self) -> trt.IPluginV3:
        cloned_plugin = AddScalarPlugin(0.0)
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    def configure_plugin(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> None:
        # This runs before `get_valid_tactics`, which is what makes per-format tactics possible
        self.current_type = dptd_in[0].desc.type
        return

    def get_output_data_types(self, input_types: List[trt.DataType]) -> List[trt.DataType]:
        return [input_types[0]]

    def get_output_shapes(self, inputs: List[trt.DimsExprs], shape_inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> List[trt.DimsExprs]:
        return [trt.DimsExprs(inputs[0])]

    def supports_format_combination(self, pos: int, in_out: List[trt.DynamicPluginTensorDesc], num_inputs: int) -> bool:
        desc = in_out[pos].desc
        if pos == 0:
            return desc.type in [trt.float32, trt.float16] and desc.format == trt.TensorFormat.LINEAR
        return desc.type == in_out[0].desc.type and desc.format == trt.TensorFormat.LINEAR

    def get_workspace_size(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> int:
        return 0

    def get_valid_tactics(self) -> List[int]:
        """Every tactic returned here gets timed by the builder.

        Returning a single-element list is what the other examples in this directory do
        (they return `[1]`), and it means "no choice to make".
        """
        if PER_FORMAT_TACTIC and self.current_type == trt.float16:
            return [TACTIC_TRITON]  # Pretend the Torch path is unsuitable for float16
        return [TACTIC_TORCH, TACTIC_TRITON]

    def set_tactic(self, tactic: int) -> None:
        """Called many times while building (once per timing trial) and once at run time."""
        self.tactic = tactic
        tactic_log.append((tactic, str(self.current_type)))
        return

    def on_shape_change(self, ptd_in: List[trt.PluginTensorDesc], ptd_out: List[trt.PluginTensorDesc]) -> None:
        return None

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc], inputs: List[int], outputs: List[int], workspace: int, stream: int) -> None:
        data_type = trt.nptype(input_desc[0].type)
        dims = input_desc[0].dims
        n_element = int(np.prod(dims))
        buffer_size = n_element * np.dtype(data_type).itemsize
        c_data_type = ctypes.c_int16 if data_type == np.float16 else ctypes.c_float

        p_input = ctypes.cast(inputs[0], ctypes.POINTER(c_data_type * n_element))[0]
        p_input = torch.as_tensor(np.ndarray(dims, dtype=data_type, buffer=p_input), device="cuda")

        if self.tactic == TACTIC_TORCH:
            p_output = p_input + self.scalar
        elif self.tactic == TACTIC_TRITON:
            p_raw = ctypes.cast(outputs[0], ctypes.POINTER(c_data_type * n_element))[0]
            p_output = torch.as_tensor(np.ndarray(dims, dtype=data_type, buffer=p_raw), device="cuda")
            block_size = 256
            add_scalar_triton_kernel[(ceil_divide(n_element, block_size), )](p_input, p_output, float(self.scalar), n_element, BLOCK_SIZE=block_size)
        else:
            raise RuntimeError(f"Unknown tactic {self.tactic}")

        cudart.cudaMemcpyAsync(outputs[0], p_output.data_ptr(), buffer_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream)
        return

    def attach_to_context(self, resource_context: trt.IPluginResourceContext) -> trt.IPluginV3:
        return self.clone()

    def get_fields_to_serialize(self) -> trt.PluginFieldCollection:
        return trt.PluginFieldCollection([trt.PluginField("scalar", np.array(self.scalar, dtype=np.float32), trt.PluginFieldType.FLOAT32)])

class AddScalarPluginCreator(trt.IPluginCreatorV3One):

    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = "AddScalarMultiTactic"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection([trt.PluginField("scalar", np.array([]), trt.PluginFieldType.FLOAT32)])
        return

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection, phase: trt.TensorRTPhase):
        value = 0.0
        for field in field_collection:
            if field.name == "scalar":
                value = float(field.data[0])
        return AddScalarPlugin(value)

def test_case(b_fp16: bool) -> None:
    """Build from scratch (so the builder really autotunes) and run once."""
    data = dict(input_data)
    trt_datatype = trt.float32
    if b_fp16:
        data["inputT0"] = input_data["inputT0"].astype(np.float16)
        trt_datatype = trt.float16

    tactic_log.clear()
    tw = TRTWrapperV1()
    plugin_creator = trt.get_plugin_registry().get_creator("AddScalarMultiTactic", "1", "")
    field_collection = trt.PluginFieldCollection([trt.PluginField("scalar", np.array(scalar, dtype=np.float32), trt.PluginFieldType.FLOAT32)])
    plugin = plugin_creator.create_plugin("AddScalarMultiTactic", field_collection, trt.TensorRTPhase.BUILD)

    input_tensor = tw.network.add_input("inputT0", trt_datatype, [-1, -1, -1])
    tw.profile.set_shape(input_tensor.name, [1, 1, 1], shape, shape)
    layer = tw.network.add_plugin_v3([input_tensor], [], plugin)
    tensor = layer.get_output(0)
    tensor.name = "outputT0"
    tw.build([tensor])

    build_log = list(tactic_log)
    offered = sorted({tactic for tactic, _ in build_log})
    print(f"    [{'float16' if b_fp16 else 'float32'}] tactics timed during build: {[TACTIC_NAME[t] for t in offered]} ({len(build_log)} set_tactic calls)")

    tactic_log.clear()
    tw.setup(data)
    tw.infer(b_print_io=False)
    chosen = sorted({tactic for tactic, _ in tactic_log})
    print(f"    [{'float16' if b_fp16 else 'float32'}] tactic baked into the engine: {[TACTIC_NAME[t] for t in chosen]}")

    if b_fp16 and PER_FORMAT_TACTIC:
        # Measured, and worth knowing: when `get_valid_tactics` returns exactly one entry the
        # builder does **not** time it at all -- `set_tactic` is called zero times during the
        # build, and only once at run time with that single candidate. So the per-format
        # restriction does not just narrow the search, it removes the timing step entirely.
        assert offered == [], f"float16 has one candidate, so nothing should have been timed, got {offered}"
        assert chosen == [TACTIC_TRITON], f"float16 should run Triton, got {chosen}"
    else:
        assert offered == [TACTIC_TORCH, TACTIC_TRITON], f"float32 should have timed both backends, got {offered}"
        assert len(chosen) == 1, f"exactly one tactic should be baked into the engine, got {chosen}"

    check_array(tw.buffer["outputT0"][0], add_scalar_cpu(data, scalar)["outputT0"], True)
    return

if __name__ == "__main__":
    for trt_path in Path(".").glob("model-multi_tactic*.trt"):
        trt_path.unlink(missing_ok=True)

    plugin_registry = trt.get_plugin_registry()
    my_plugin_creator = AddScalarPluginCreator()
    if my_plugin_creator.name not in [creator.name for creator in plugin_registry.all_creators]:
        plugin_registry.register_creator(my_plugin_creator, "")

    test_case(False)  # float32: both backends compete
    test_case(True)  # float16: per-format restriction leaves only Triton

    print("\nFinish")
