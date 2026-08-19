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

# An `IPluginV3` whose kernel is written in CuteDSL, CUTLASS's Python DSL.
#
# The other Python-plugin examples (`05-Plugin/PythonPlugin`) get their kernel from NVRTC, CuPy,
# Numba, Triton or PyTorch. CuteDSL is a different point in that space: the kernel is Python, but it
# is compiled through CUTLASS and has direct access to shared memory, warp intrinsics and CuTe
# layouts - so it can express the kind of block-level reduction a real LLM kernel needs.
#
# The operator is RMSNorm, the normalization used by essentially every modern LLM. What this example
# is really about is the three-way hand-off in `enqueue()`:
#
#     raw device pointer from TensorRT
#       -> cupy.cuda.UnownedMemory   (wrap a foreign pointer without owning it)
#       -> torch.as_tensor           (via __cuda_array_interface__)
#       -> cute.runtime.from_dlpack  (via __dlpack__, gives the cute.Tensor the kernel sees)
#
# Every hop is zero-copy - the same GPU bytes are just re-typed. The first hop is
# `tensorrt_cookbook.wrap_device_pointer`, shared with `05-Plugin/PythonPlugin`.
#
# Requirements: `pip install nvidia-cutlass-dsl` and an Ampere (SM80) or newer GPU.

from pathlib import Path
from typing import List

try:
    import cupy as cp  # noqa: F401  (imported for its CUDA runtime init, used via wrap_device_pointer)
    import cutlass
    import cutlass.cute as cute
except Exception as e:  # optional packages, see this directory's README
    print(f"[SKIP] CuteDSL stack unusable ({type(e).__name__}: {e});"
          " pip install nvidia-cutlass-dsl cupy-cuda12x")
    raise SystemExit(0)
import numpy as np
import tensorrt as trt
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from tensorrt_cookbook import TRTWrapperV1, case_mark, check_array, wrap_device_pointer

trt_file = Path("model.trt")

THREADS_PER_BLOCK = 256
HIDDEN_DIM = 1024
EPSILON = 1.0e-5

def rms_norm_cpu(x, weight, epsilon):
    x32 = x.astype(np.float32)
    scale = 1.0 / np.sqrt((x32 * x32).mean(axis=-1, keepdims=True) + epsilon)
    return (x32 * weight.astype(np.float32) * scale).astype(x.dtype)

# ==================================================================================================
# The CuteDSL kernel
# ==================================================================================================

@cute.kernel
def rms_norm_kernel(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mY: cute.Tensor,
    threads_per_block: cutlass.Constexpr,
    hidden_dim: cutlass.Constexpr,
    epsilon: cutlass.Constexpr,
):
    # One block per token. The threads of a block cooperatively sum the squares across the hidden
    # dimension into shared memory, reduce that to a single RMS value, then rescale.
    smem = cutlass.utils.SmemAllocator()
    sdata = smem.allocate_tensor(cutlass.Float32, layout=cute.make_layout(threads_per_block), byte_alignment=16)
    rms = smem.allocate_tensor(cutlass.Float32, layout=cute.make_layout(1))

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Accumulate in FP32 whatever the input dtype is: with hidden_dim in the thousands, summing
    # squares in FP16 loses too much precision.
    local_sum = cutlass.Float32(0.0)
    for i in cutlass.range(tidx, hidden_dim, threads_per_block):
        x = cutlass.Float32(mX[bidx, i])
        local_sum += x * x
    sdata[tidx] = local_sum
    cute.arch.sync_threads()

    # Tree reduction in shared memory down to one warp, then a warp shuffle.
    if tidx < 128:
        sdata[tidx] += sdata[tidx + 128]
    cute.arch.sync_threads()

    if tidx < 64:
        sdata[tidx] += sdata[tidx + 64]
    cute.arch.sync_threads()

    if tidx < 32:
        v = sdata[tidx] + sdata[tidx + 32]
        v = cute.arch.warp_reduction_sum(v, threads_in_group=32)
        if tidx == 0:
            rms[0] = cute.math.rsqrt(v / hidden_dim + epsilon, fastmath=True)
    cute.arch.sync_threads()

    scale = rms[0]
    for i in cutlass.range(tidx, hidden_dim, threads_per_block):
        y = cutlass.Float32(mX[bidx, i]) * cutlass.Float32(mW[i]) * scale
        mY[bidx, i] = y.to(mY.element_type)  # Store back in the output's dtype (FP16)

@cute.jit
def rms_norm_launch(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mY: cute.Tensor,
    num_tokens: cutlass.Int32,
    hidden_dim: cutlass.Constexpr,
    epsilon: cutlass.Constexpr,
    stream: CUstream,
):
    # `num_tokens` is a *runtime* value (it only sets the grid dimension) while `hidden_dim` is
    # `Constexpr` and baked in. That is what lets one compiled kernel serve every sequence length
    # the optimization profile allows.
    # The `stream` parameter is what makes CuteDSL launch on TensorRT's stream: the DSL runtime
    # picks up the `CUstream`-typed argument, it is not forwarded to `.launch()` explicitly.
    rms_norm_kernel(mX, mW, mY, THREADS_PER_BLOCK, hidden_dim, epsilon).launch(
        grid=(num_tokens, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
    )

# ==================================================================================================
# The plugin
# ==================================================================================================

class RmsNormPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):

    def __init__(self, epsilon: float = EPSILON):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)
        self.plugin_name = "RmsNorm"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.num_outputs = 1
        self.epsilon = epsilon
        # JIT cache keyed by (hidden_dim, epsilon): both are `Constexpr`, so changing either
        # produces a different binary. `num_tokens` is not part of the key.
        self.compiled_dict = {}

    def get_capability_interface(self, plugin_capability_type: trt.PluginCapabilityType) -> trt.IPluginCapability:
        return self

    def get_output_data_types(self, input_types: List[trt.DataType]) -> List[trt.DataType]:
        return [input_types[0]]

    def get_output_shapes(self, inputs: List[trt.DimsExprs], shape_inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> List[trt.DimsExprs]:
        return [trt.DimsExprs(inputs[0])]

    def supports_format_combination(self, pos: int, in_out: List[trt.DynamicPluginTensorDesc], num_inputs: int) -> bool:
        desc = in_out[pos].desc
        if desc.format != trt.TensorFormat.LINEAR:
            return False
        if pos == 1:  # `weight` stays FP32 so the scale is not quantized twice
            return desc.type == trt.float32
        return desc.type == trt.float16  # `x` and `y`

    def configure_plugin(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> None:
        return

    def get_workspace_size(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> int:
        return 0

    def get_valid_tactics(self) -> List[int]:
        return [1]

    def set_tactic(self, tactic: int) -> None:
        return

    def on_shape_change(self, ptd_in: List[trt.PluginTensorDesc], ptd_out: List[trt.PluginTensorDesc]) -> None:
        # Only `num_tokens` can change, and it is not part of the JIT cache key, so nothing to do.
        return

    def attach_to_context(self, resource_context: trt.IPluginResourceContext) -> trt.IPluginV3:
        return self.clone()

    def get_fields_to_serialize(self) -> trt.PluginFieldCollection:
        return trt.PluginFieldCollection([trt.PluginField("epsilon", np.array([self.epsilon], dtype=np.float32), trt.PluginFieldType.FLOAT32)])

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc], inputs: List[int], outputs: List[int], workspace: int, stream: int) -> None:
        num_tokens, hidden_dim = (int(d) for d in input_desc[0].dims)

        # Zero-copy: cupy wraps the pointer TensorRT owns, torch reads cupy's
        # __cuda_array_interface__, CuteDSL reads torch's __dlpack__ capsule.
        x_tensor = torch.as_tensor(wrap_device_pointer(inputs[0], (num_tokens, hidden_dim), trt.nptype(input_desc[0].type), self), device="cuda")
        w_tensor = torch.as_tensor(wrap_device_pointer(inputs[1], (hidden_dim, ), trt.nptype(input_desc[1].type), self), device="cuda")
        y_tensor = torch.as_tensor(wrap_device_pointer(outputs[0], (num_tokens, hidden_dim), trt.nptype(output_desc[0].type), self), device="cuda")

        mX = from_dlpack(x_tensor, assumed_align=16)
        mW = from_dlpack(w_tensor, assumed_align=16)
        mY = from_dlpack(y_tensor, assumed_align=16)

        key = (hidden_dim, self.epsilon)
        if key not in self.compiled_dict:
            # `make_fake_stream()` is a compile-time placeholder; the real stream is passed below.
            print(f"[CuteDSL] JIT-compiling for {key = } (num_tokens = {num_tokens} is not part of the key)")
            self.compiled_dict[key] = cute.compile(rms_norm_launch, mX, mW, mY, num_tokens, hidden_dim, self.epsilon, make_fake_stream())
        self.compiled_dict[key](mX, mW, mY, num_tokens, CUstream(stream))

    def clone(self) -> trt.IPluginV3:
        cloned = RmsNormPlugin(self.epsilon)
        cloned.__dict__.update(self.__dict__)
        cloned.compiled_dict = {}  # Each execution context gets its own cubins
        return cloned

class RmsNormPluginCreator(trt.IPluginCreatorV3One):

    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = "RmsNorm"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection([trt.PluginField("epsilon", np.array([], dtype=np.float32), trt.PluginFieldType.FLOAT32)])

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection, phase: trt.TensorRTPhase):
        epsilon = EPSILON
        for field in field_collection:
            if field.name == "epsilon":
                epsilon = float(field.data[0])
        return RmsNormPlugin(epsilon)

# ==================================================================================================

def make_data(num_tokens):
    np.random.seed(31193)
    return {
        "x": np.random.normal(0, 1, (num_tokens, HIDDEN_DIM)).astype(np.float16),
        "weight": np.random.normal(1, 0.1, (HIDDEN_DIM, )).astype(np.float32),
    }

def build_engine():
    tw = TRTWrapperV1(trt_file=trt_file)
    if tw.engine_bytes is None:
        plugin_creator = trt.get_plugin_registry().get_creator("RmsNorm", "1", "")
        field_collection = trt.PluginFieldCollection([trt.PluginField("epsilon", np.array([EPSILON], dtype=np.float32), trt.PluginFieldType.FLOAT32)])
        plugin = plugin_creator.create_plugin("RmsNorm", field_collection, trt.TensorRTPhase.BUILD)

        # `num_tokens` is dynamic, `hidden_dim` is static - which is exactly the split the kernel
        # was written for (runtime grid dimension vs. compile-time Constexpr).
        x = tw.network.add_input("x", trt.float16, [-1, HIDDEN_DIM])
        weight = tw.network.add_input("weight", trt.float32, [HIDDEN_DIM])
        tw.profile.set_shape(x.name, [1, HIDDEN_DIM], [128, HIDDEN_DIM], [512, HIDDEN_DIM])

        layer = tw.network.add_plugin_v3([x, weight], [], plugin)
        tensor = layer.get_output(0)
        tensor.name = "y"

        tw.build([tensor])
        tw.serialize_engine(trt_file)
    return tw

def run_rms_norm(tw, num_tokens):
    data = make_data(num_tokens)
    tw.setup(data, b_print_io=False)
    tw.infer(b_print_io=False)

    # FP16 store plus a fast rsqrt, so compare with a tolerance: the observed error is about one
    # FP16 ULP (2^-9 for values in [1, 2)), not an algorithmic difference.
    check_array(tw.buffer["y"][0], rms_norm_cpu(data["x"], data["weight"], EPSILON), True, error_epsilon=5e-3)

@case_mark
def case_dynamic_shape():
    """Run two different `num_tokens` through one engine and one plugin instance.

    The `[CuteDSL] JIT-compiling` line appears only once: the cache is keyed on
    (hidden_dim, epsilon), both `Constexpr`, while `num_tokens` only sets the grid dimension. That
    split is the reason a dynamic sequence length costs nothing at the DSL level.
    """
    tw = build_engine()
    for num_tokens in [128, 512]:
        print(f"--- {num_tokens = }")
        run_rms_norm(tw, num_tokens)

if __name__ == "__main__":
    trt_file.unlink(missing_ok=True)

    plugin_registry = trt.get_plugin_registry()
    my_plugin_creator = RmsNormPluginCreator()
    if my_plugin_creator.name not in [creator.name for creator in plugin_registry.all_creators]:
        plugin_registry.register_creator(my_plugin_creator, "")

    case_dynamic_shape()

    print("Finish")
