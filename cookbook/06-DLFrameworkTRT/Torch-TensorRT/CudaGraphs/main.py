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
"""Replaying a compiled module as a CUDA graph instead of launching kernel by kernel.

`torch_tensorrt.runtime.enable_cudagraphs(module)` records the module's kernel
launches once and replays the recording afterwards, so the CPU issues one graph
launch instead of N kernel launches.

That only helps if launching *was* the bottleneck. `case_when_it_helps` measures
it across four shapes and the answer ranges from **1.01x (nothing)** to **2.2x**:
a ResNet that TensorRT already fused into a few long kernels has no launch
overhead left to remove, while a stack of small layers is almost all overhead.

The recording is tied to the input shape, and only one recording is kept --
`case_shape_change` shows what that costs.
"""

import time

import torch
import torch_tensorrt
import torchvision.models as models

from tensorrt_cookbook import case_mark

torch.manual_seed(31193)

N_WARMUP, N_TEST = 50, 300
N_LAYER, N_FEATURE = 40, 64

class ManySmallOps(torch.nn.Module):
    """Many short kernels, i.e. the launch-bound extreme."""

    def __init__(self, n_layer: int = N_LAYER, n_feature: int = N_FEATURE) -> None:
        """Build a stack of small `Linear` layers."""
        super().__init__()
        self.layer = torch.nn.ModuleList([torch.nn.Linear(n_feature, n_feature) for _ in range(n_layer)])

    def forward(self, x):
        """Forward pass."""
        for layer in self.layer:
            x = torch.relu(layer(x))
        return x

class TinyModel(torch.nn.Module):
    """Three element-wise operators, used to force a graph break."""

    def forward(self, x):
        """Forward pass."""
        return torch.relu((x + 2) * 0.5)

def benchmark(module, data) -> float:
    """Mean latency in milliseconds."""
    with torch.no_grad():
        for _ in range(N_WARMUP):
            module(data)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_TEST):
            module(data)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / N_TEST

def compile_static(model, data, **kwargs):
    """Compile for one fixed shape."""
    return torch_tensorrt.compile(model, ir="dynamo", inputs=[data], min_block_size=1, **kwargs)

@case_mark
def case_three_ways_to_enable() -> None:
    """The context manager, the session switch, and off -- all give the same numbers.

    The context manager is the one to prefer: it scopes the change and hands back
    the module to call. `set_cudagraphs_mode` flips it for the whole session,
    which is convenient and easy to forget about.
    """
    model = ManySmallOps().cuda().eval()
    data = torch.randn(1, N_FEATURE).cuda()
    compiled = compile_static(model, data)

    with torch.no_grad():
        reference = model(data)

        with torch_tensorrt.runtime.enable_cudagraphs(compiled) as cudagraphs_module:
            by_context = cudagraphs_module(data)

        torch_tensorrt.runtime.set_cudagraphs_mode(True)
        by_session = compiled(data)
        torch_tensorrt.runtime.set_cudagraphs_mode(False)
        plain = compiled(data)

    for tag, output in [("context manager  ", by_context), ("set_cudagraphs_mode", by_session), ("off              ", plain)]:
        print(f"    {tag}: max |eager - TensorRT| = {(output - reference).abs().max().item():.1e}")
    return

@case_mark
def case_when_it_helps() -> None:
    """Whether CUDA graphs pay off is a property of the model, not a setting.

    The recording removes *launch* overhead. A model whose kernels are long
    enough to hide their own launches has nothing to gain -- and that includes
    the ResNet the upstream tutorial uses.
    """
    print(f"    {'model':<32}{'plain':>10}{'cudagraphs':>12}{'gain':>8}")
    for tag, model, shape in [
        ("resnet18, batch 16", models.resnet18(weights=None).cuda().eval(), (16, 3, 224, 224)),
        ("resnet18, batch 1", models.resnet18(weights=None).cuda().eval(), (1, 3, 224, 224)),
        (f"{N_LAYER} small Linear, batch 1", ManySmallOps().cuda().eval(), (1, N_FEATURE)),
        (f"{N_LAYER} small Linear, batch 128", ManySmallOps().cuda().eval(), (128, N_FEATURE)),
    ]:
        data = torch.randn(*shape).cuda()
        compiled = compile_static(model, data)
        plain_ms = benchmark(compiled, data)
        with torch.no_grad(), torch_tensorrt.runtime.enable_cudagraphs(compiled) as cudagraphs_module:
            cudagraphs_module(data)  # Record before timing
            graph_ms = benchmark(cudagraphs_module, data)
        print(f"    {tag:<32}{plain_ms:>9.3f} ms{graph_ms:>10.3f} ms{plain_ms / graph_ms:>7.2f}x")
    print("    long kernels leave no launch overhead to remove; short ones are almost all overhead")
    return

@case_mark
def case_shape_change() -> None:
    """A recording belongs to one input shape, and only one is kept.

    Feeding a new shape re-records. Feeding the *old* shape back re-records
    again -- there is no per-shape cache. So a workload that alternates between
    two shapes pays the recording cost on every call and is better off with
    CUDA graphs disabled.
    """
    model = ManySmallOps().cuda().eval()
    compiled = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=torch_tensorrt.Input(min_shape=(1, N_FEATURE), opt_shape=(64, N_FEATURE), max_shape=(256, N_FEATURE), dtype=torch.float32, name="x"),
        min_block_size=1,
    )
    with torch.no_grad(), torch_tensorrt.runtime.enable_cudagraphs(compiled) as cudagraphs_module:
        for batch, note in [(64, "first call, records"), (64, "replay"), (128, "new shape, re-records"), (128, "replay"), (64, "back to 64, re-records again")]:
            data = torch.randn(batch, N_FEATURE).cuda()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cudagraphs_module(data)
            torch.cuda.synchronize()
            print(f"      batch={batch:<4}: {(time.perf_counter() - t0) * 1000:7.3f} ms   {note}")
    return

@case_mark
def case_graph_break() -> None:
    """With graph breaks the context returns a wrapper that records everything.

    `torch_executed_ops` splits the module into engine / torch / engine here.
    Without CUDA graphs each piece is launched separately; the wrapper records
    the whole sequence, torch segment included, so the breaks stop costing extra
    launches.
    """
    model = TinyModel().cuda().eval()
    data = torch.randn(1, 3, 224, 224).cuda()
    compiled = compile_static(model, data, pass_through_build_failures=True, torch_executed_ops={"torch.ops.aten.mul.Tensor"})

    segment = [node.name for node in compiled.graph.nodes if node.op == "call_module"]
    print(f"    segments: {segment}")

    plain_ms = benchmark(compiled, data)
    with torch.no_grad(), torch_tensorrt.runtime.enable_cudagraphs(compiled) as cudagraphs_module:
        print(f"    context returned {type(cudagraphs_module).__name__}, not the module itself")
        cudagraphs_module(data)
        graph_ms = benchmark(cudagraphs_module, data)
    print(f"    plain {plain_ms:.3f} ms, cudagraphs {graph_ms:.3f} ms ({plain_ms / graph_ms:.2f}x)")
    return

if __name__ == "__main__":
    case_three_ways_to_enable()
    case_when_it_helps()
    case_shape_change()
    case_graph_break()

    print("\nFinish")
