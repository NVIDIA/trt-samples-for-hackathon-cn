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
"""One plugin, three kernels, no new plugin class per kernel.

Run `python3 main.py` and compare with `05-Plugin/QuickDeployablePlugin/`, which registers one
plugin per operation. Here `cookbook::byok` is registered once and the operation is chosen by a
string attribute.
"""

from typing import List, Tuple

import numpy as np
import tensorrt as trt
import tensorrt.plugin as trtp
import torch
import triton
import triton.language as tl
from kernel_registry import KERNEL_REGISTRY, get_kernel, make_spec, parse_spec, register_kernel

from tensorrt_cookbook import TRTWrapperV1, case_mark

# ------------------------------------------------------------------------------------------------
# The kernels. Each is a plain function; registering one is a single decorator.
# ------------------------------------------------------------------------------------------------

@triton.jit
def _add_mul_kernel(a_ptr, b_ptr, out_ptr, n_element, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_element
    a = tl.load(a_ptr + offset, mask=mask)
    b = tl.load(b_ptr + offset, mask=mask)
    tl.store(out_ptr + offset, (a + b) * alpha, mask=mask)

@register_kernel("cookbook.residual_add")
def residual_add(a, b, out, extra_args, stream):
    """out = a + b, elementwise (Triton)."""
    n = a.numel()
    block = 256
    _add_mul_kernel[(triton.cdiv(n, block), )](a, b, out, n, 1.0, BLOCK_SIZE=block)

@register_kernel("cookbook.scaled_add")
def scaled_add(a, b, out, extra_args, stream):
    """out = (a + b) * alpha, alpha from the spec (Triton)."""
    alpha = float(extra_args.get("alpha", 1.0))
    n = a.numel()
    block = 256
    _add_mul_kernel[(triton.cdiv(n, block), )](a, b, out, n, alpha, BLOCK_SIZE=block)

@register_kernel("cookbook.row_sum", rank_delta=-1)
def row_sum(a, b, out, extra_args, stream):
    """out = sum(a + b) over the last axis, so the output has one fewer dimension (torch)."""
    torch.sum(a + b, dim=-1, out=out)

# ------------------------------------------------------------------------------------------------
# The one plugin. Registered once, for every kernel above and every kernel added later.
# ------------------------------------------------------------------------------------------------

@trtp.register("cookbook::byok")
def byok_desc(inp0: trtp.TensorDesc, inp1: trtp.TensorDesc, spec: str) -> trtp.TensorDesc:
    kernel_name, _ = parse_spec(spec)
    rank_delta = get_kernel(kernel_name)["rank_delta"]
    if rank_delta == 0:
        return inp0.like()
    # A reduction: drop `-rank_delta` trailing dimensions. The shape is built from symbolic shape
    # expressions, not from concrete numbers, because this runs during shape inference where the
    # batch dimension is still -1.
    shape_expr = inp0.shape_expr
    kept = tuple(shape_expr[i] for i in range(len(shape_expr) + rank_delta))
    return trtp.from_shape_expr(kept, dtype=inp0.dtype)

@trtp.autotune("cookbook::byok")
def byok_autotune(inp0: trtp.TensorDesc, inp1: trtp.TensorDesc, spec: str, outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
    return [trtp.AutoTuneCombination("FP32, FP32, FP32")]

@trtp.impl("cookbook::byok")
def byok_impl(inp0: trtp.Tensor, inp1: trtp.Tensor, spec: str, outputs: Tuple[trtp.Tensor], stream: int) -> None:
    kernel_name, extra_args = parse_spec(spec)
    launch = get_kernel(kernel_name)["launch"]
    a = torch.as_tensor(inp0, device="cuda")
    b = torch.as_tensor(inp1, device="cuda")
    out = torch.as_tensor(outputs[0], device="cuda")
    # One dispatch for every kernel that will ever be registered. This line is the whole feature.
    launch(a, b, out, extra_args, stream)

# ------------------------------------------------------------------------------------------------

def run(kernel_name: str, shape: list, **extra_args) -> np.ndarray:
    """Build and run a one-layer network whose only layer is the shared plugin."""
    rng = np.random.default_rng(31193)
    data = {
        "a": rng.random(shape, dtype=np.float32),
        "b": rng.random(shape, dtype=np.float32),
    }
    tw = TRTWrapperV1()
    tensor_a = tw.network.add_input("a", trt.float32, [-1] + shape[1:])
    tensor_b = tw.network.add_input("b", trt.float32, [-1] + shape[1:])
    tw.profile.set_shape(tensor_a.name, [1] + shape[1:], shape, shape)
    tw.profile.set_shape(tensor_b.name, [1] + shape[1:], shape, shape)

    spec = make_spec(kernel_name, **extra_args)
    layer = tw.network.add_plugin(trtp.op.cookbook.byok(tensor_a, tensor_b, spec=spec))

    tw.build([layer.get_output(0)])
    tw.setup(data, b_print_io=False)
    tw.infer(b_print_io=False)
    return tw.buffer[tw.tensor_name_list[-1]][0], data

@case_mark
def case_one_plugin_three_kernels():
    """The same registered plugin produces three different operations."""
    print(f"    Registered kernels: {sorted(KERNEL_REGISTRY)}")
    shape = [2, 3, 8]

    got, data = run("cookbook.residual_add", shape)
    expected = data["a"] + data["b"]
    print(f"    residual_add -> shape {got.shape}, max abs diff {np.abs(got - expected).max():.3e}")
    assert np.allclose(got, expected, atol=1e-5)

    got, data = run("cookbook.scaled_add", shape, alpha=3.0)
    expected = (data["a"] + data["b"]) * 3.0
    print(f"    scaled_add   -> shape {got.shape}, max abs diff {np.abs(got - expected).max():.3e} (alpha came from the spec)")
    assert np.allclose(got, expected, atol=1e-5)

    got, data = run("cookbook.row_sum", shape)
    expected = (data["a"] + data["b"]).sum(axis=-1)
    print(f"    row_sum      -> shape {got.shape}, max abs diff {np.abs(got - expected).max():.3e} (rank dropped by 1)")
    assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
    assert np.allclose(got, expected, atol=1e-4)

@case_mark
def case_adding_a_kernel_is_one_decorator():
    """No new plugin, no new creator, no re-registration: just a function."""
    before = set(KERNEL_REGISTRY)

    @register_kernel("cookbook.max_of_two")
    def max_of_two(a, b, out, extra_args, stream):
        """out = max(a, b), elementwise (torch)."""
        torch.maximum(a, b, out=out)

    added = sorted(set(KERNEL_REGISTRY) - before)
    print(f"    Added {added} with one decorator; plugin registrations unchanged")

    shape = [2, 4, 6]
    got, data = run("cookbook.max_of_two", shape)
    expected = np.maximum(data["a"], data["b"])
    print(f"    max_of_two   -> shape {got.shape}, max abs diff {np.abs(got - expected).max():.3e}")
    assert np.allclose(got, expected, atol=1e-5)

@case_mark
def case_unknown_kernel_fails_at_build_time():
    """An unregistered name is rejected while building, not inside enqueue.

    `make_spec` looks the kernel up before the attribute is ever attached to a layer. Catching it
    here means the error names the typo and the known kernels; catching it in `enqueue` would mean
    a plugin exception in the middle of inference.
    """
    try:
        make_spec("cookbook.does_not_exist")
        raise AssertionError("Expected a KeyError")
    except KeyError as e:
        print(f"    {e}")

if __name__ == "__main__":
    case_one_plugin_three_kernels()
    case_adding_a_kernel_is_one_decorator()
    case_unknown_kernel_fails_at_build_time()

    print("\nFinish")
