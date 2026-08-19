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
"""Running a custom Triton kernel inside the TensorRT engine, via a QDP plugin.

A custom `torch.library` op has no TensorRT converter, so the partitioner leaves
it to PyTorch and the graph is cut into `engine / torch / engine`. The TensorRT
10.7+ Quick Deployable Plugin system plus `generate_plugin` /
`generate_plugin_converter` removes the cut: the kernel becomes a plugin layer
inside a single engine.

**It does not follow that this is faster.** The generated plugin is a *JIT*
plugin -- at engine runtime TensorRT calls back into Python to run the Triton
kernel through PyTorch. Here that callback costs more than the graph break it
removed, and the single-engine version is **2.5x slower**. See
`case_measure_the_tradeoff`; AOT plugins are the way out.

Two things are needed before any of this works:

    @torch.library.custom_op   makes the kernel a first-class PyTorch operator
    @torch.library.register_fake  the meta kernel: shapes and dtypes only, no
                               compute. `torch.export` and `generate_plugin`
                               both drive it symbolically, so it must be right.
"""

import time

import torch
import torch_tensorrt
import triton
import triton.language as tl

from tensorrt_cookbook import case_mark

torch.manual_seed(31193)

SHAPE = (64, 64)
BLOCK_SIZE = 1024
N_WARMUP, N_TEST = 50, 300

@triton.jit
def scale_mul_kernel(X, Y, Z, a, b, BLOCK_SIZE: tl.constexpr):
    """`Z = X * Y * a + b`, one block per `BLOCK_SIZE` elements."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(Z + offset, tl.load(X + offset) * tl.load(Y + offset) * a + b)

@torch.library.custom_op("cookbook::scale_mul", mutates_args=())
def scale_mul(X: torch.Tensor, Y: torch.Tensor, b: float = 0.2, a: int = 2) -> torch.Tensor:
    """The eager implementation, which is also what the JIT plugin calls back into."""
    assert X.is_cuda and Y.is_cuda, "tensors must be on the GPU"
    assert X.shape == Y.shape, "tensors must have the same shape"
    Z = torch.empty_like(X)
    scale_mul_kernel[lambda meta: (X.numel() // meta["BLOCK_SIZE"], )](X, Y, Z, a, b, BLOCK_SIZE=BLOCK_SIZE)
    return Z

@torch.library.register_fake("cookbook::scale_mul")
def _(x: torch.Tensor, y: torch.Tensor, b: float = 0.2, a: int = 2) -> torch.Tensor:
    """Meta kernel: element-wise, so the output matches the first input."""
    return x

class Model(torch.nn.Module):
    """TensorRT-native ops on both sides of the custom one, to expose the cut."""

    def forward(self, x, y):
        """Forward pass."""
        z = torch.add(x, y)
        w = torch.ops.cookbook.scale_mul.default(x, z, b=0.5)
        return torch.relu(w)

model = Model().cuda().eval()
data = (
    torch.randint(0, 5, SHAPE, device="cuda", dtype=torch.float),
    torch.randint(0, 5, SHAPE, device="cuda", dtype=torch.float),
)

def segment_of(module) -> list:
    """The partitioner's segments, `_run_on_gpu_*` being the PyTorch ones."""
    return [node.name for node in module.graph.nodes if node.op == "call_module"]

def benchmark(module) -> float:
    """Mean latency in milliseconds."""
    with torch.no_grad():
        for _ in range(N_WARMUP):
            module(*data)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_TEST):
            module(*data)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / N_TEST

def compile_and_check(tag: str):
    """Compile, verify against eager, and report the segmentation."""
    compiled = torch_tensorrt.compile(model, inputs=list(data), min_block_size=1)
    with torch.no_grad():
        difference = (compiled(*data) - model(*data)).abs().max().item()
    print(f"    {tag:<26}: segments {segment_of(compiled)}")
    print(f"    {'':<26}  max |eager - TensorRT| = {difference:.1e}, {benchmark(compiled):.3f} ms")
    assert difference == 0.0, "the custom kernel changed the result"
    return compiled

@case_mark
def case_without_plugin() -> None:
    """No converter for the custom op, so the partitioner cuts around it.

    `_run_on_gpu_1` is the custom kernel, executed by PyTorch between two
    engines. Correct, but every call crosses the TensorRT/PyTorch boundary
    twice.
    """
    global without_plugin
    without_plugin = compile_and_check("no plugin")
    return

@case_mark
def case_with_plugin() -> None:
    """`generate_plugin` + `generate_plugin_converter` collapse it to one engine.

    `generate_plugin` drives the meta kernel under `FakeTensorMode` to derive a
    symbolic shape descriptor, and wraps the eager op as the plugin's runtime
    implementation. `generate_plugin_converter` then registers a
    `dynamo_tensorrt_converter` that emits the plugin layer.

    Neither call needs the kernel source -- everything comes from the registered
    op and its meta kernel, which is why the meta kernel has to be correct.
    """
    global with_plugin
    torch_tensorrt.dynamo.conversion.plugins.generate_plugin("cookbook::scale_mul")
    torch_tensorrt.dynamo.conversion.plugins.generate_plugin_converter(
        "cookbook::scale_mul",
        supports_dynamic_shapes=True,
        requires_output_allocator=False,
    )
    with_plugin = compile_and_check("QDP plugin")
    print("    one engine: the Triton kernel is now a plugin layer inside it")
    return

@case_mark
def case_measure_the_tradeoff() -> None:
    """Fewer engines is not the same as faster.

    The generated plugin is **JIT**: at engine runtime TensorRT hands control
    back to Python, which runs the Triton kernel through PyTorch and copies the
    result into TensorRT's output buffer. That callback happens on every
    inference, whereas the graph break it replaced cost only two boundary
    crossings of an already-launched graph.

    On this model the callback is the more expensive of the two. The structural
    win is real -- one engine, no PyTorch segment, and the whole thing can now
    be captured as a CUDA graph or serialised as a single engine -- but the
    latency went the wrong way.

    The fix is an **AOT plugin** (`aot_plugin.py` upstream): the kernel is
    compiled to PTX and embedded, so there is no Python at runtime.
    """
    without_ms, with_ms = benchmark(without_plugin), benchmark(with_plugin)
    print(f"    no plugin  : {len(segment_of(without_plugin))} segments, {without_ms:.3f} ms")
    print(f"    QDP plugin : {len(segment_of(with_plugin))} segment , {with_ms:.3f} ms  ({with_ms / without_ms:.2f}x)")
    print("    measure before adopting: a JIT plugin buys structure, not speed")
    return

if __name__ == "__main__":
    case_without_plugin()
    case_with_plugin()
    case_measure_the_tradeoff()

    print("\nFinish")
