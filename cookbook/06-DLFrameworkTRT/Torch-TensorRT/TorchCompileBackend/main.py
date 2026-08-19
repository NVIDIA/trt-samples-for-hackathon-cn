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
"""AOT (`ir="dynamo"`) against JIT (`torch.compile`), and when to pick which.

Both front ends end in the same TensorRT builder, so a model that works in both
gives the same numbers. What differs is everything around the build:

    AOT  `torch_tensorrt.dynamo.compile(exported_program, ...)`
         builds during the compile call, returns a `GraphModule` you can save,
         and needs the model to survive `torch.export`.

    JIT  `torch.compile(model, backend="tensorrt")`
         builds on the first call, returns a wrapper that cannot be saved, and
         tolerates anything dynamo can trace -- including control flow that
         `torch.export` refuses outright.

`case_data_dependent_control_flow` is the one that decides the choice for a real
model: if `torch.export` cannot export it, AOT is not an option at all.

`torch_tensorrt.compile(..., ir="torch_compile")` is the same thing as the
`torch.compile` call, see `case_two_spellings`.
"""

import tempfile
import time
from pathlib import Path

import torch
import torch_tensorrt
from torch._dynamo.utils import counters

from tensorrt_cookbook import case_mark

torch.manual_seed(31193)

N_FEATURE = 64
BATCH = 8
work_path = Path(tempfile.mkdtemp(prefix="torch-trt-frontend-"))

class Simple(torch.nn.Module):
    """Exportable by anything, used wherever the model is not the point."""

    def __init__(self) -> None:
        """Build the layer."""
        super().__init__()
        self.linear = torch.nn.Linear(N_FEATURE, N_FEATURE)

    def forward(self, x):
        """Forward pass."""
        return torch.relu(self.linear(x)) + 1.0

class DataDependent(torch.nn.Module):
    """The branch depends on tensor *values*, which is what breaks `torch.export`."""

    def __init__(self) -> None:
        """Build the layer."""
        super().__init__()
        self.linear = torch.nn.Linear(N_FEATURE, N_FEATURE)

    def forward(self, x):
        """Forward pass with a data-dependent branch."""
        y = self.linear(x)
        if y.sum() > 0:
            return torch.relu(y)
        return torch.tanh(y)

data = torch.randn(BATCH, N_FEATURE).cuda()

def n_trt_engine(module) -> int:
    """How many sub-graphs became TensorRT engines (0 means nothing was converted)."""
    return sum(1 for node in module.graph.nodes if node.op == "call_module" and "_run_on_acc" in node.name)

def compile_aot(model, **kwargs):
    """AOT: export first, then build."""
    exported = torch.export.export(model, (data, ))
    return torch_tensorrt.dynamo.compile(exported, (data, ), min_block_size=1, **kwargs)

@case_mark
def case_two_spellings() -> None:
    """`torch.compile(backend=...)` and `ir="torch_compile"` are the same path.

    Two backend names are registered and both work; `"tensorrt"` is the shorter
    one used throughout this directory.
    """
    reference = Simple().cuda().eval()
    with torch.no_grad():
        expected = reference(data)

    for tag, build in [
        ("torch.compile(backend='tensorrt')      ", lambda m: torch.compile(m, backend="tensorrt", options={"min_block_size": 1})),
        ("torch.compile(backend='torch_tensorrt')", lambda m: torch.compile(m, backend="torch_tensorrt", options={"min_block_size": 1})),
        ("torch_tensorrt.compile(ir='torch_compile')", lambda m: torch_tensorrt.compile(m, ir="torch_compile", inputs=[data], min_block_size=1)),
    ]:
        torch._dynamo.reset()
        with torch.no_grad():
            output = build(reference)(data)
        print(f"    {tag}: max |eager - TensorRT| = {(expected - output).abs().max().item():.1e}")
    return

@case_mark
def case_when_is_the_engine_built() -> None:
    """The defining difference in one measurement: AOT pays up front, JIT pays late.

    Neither is cheaper overall -- the same engine gets built either way. It is a
    question of *where* the seconds land: in a build step, or in whatever request
    happens to arrive first.
    """
    model = Simple().cuda().eval()

    t0 = time.time()
    aot = compile_aot(model)
    aot_compile = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        aot(data)
    print(f"    AOT: compile call {aot_compile:5.2f} s, first inference {time.time() - t0:5.3f} s")

    torch._dynamo.reset()
    t0 = time.time()
    jit = torch.compile(model, backend="tensorrt", options={"min_block_size": 1})
    jit_compile = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        jit(data)
    print(f"    JIT: compile call {jit_compile:5.2f} s, first inference {time.time() - t0:5.3f} s")
    print("    `torch.compile` returns immediately -- it has not looked at the model yet")
    return

@case_mark
def case_data_dependent_control_flow() -> None:
    """The case where the choice is made for you.

    `if y.sum() > 0` branches on a *value*, so `torch.export` cannot pick a
    branch and cannot represent both. It fails with `GuardOnDataDependentSymNode`,
    which rules out the whole AOT path -- there is no `ExportedProgram` to compile.

    Dynamo does not have to decide: it compiles one graph up to the branch,
    evaluates the condition in Python, and compiles the continuation separately.
    Two frames, both handed to TensorRT, and the result is still exact.
    """
    model = DataDependent().cuda().eval()

    try:
        torch.export.export(model, (data, ))
        print("    AOT torch.export: unexpectedly succeeded")
    except Exception as e:
        print(f"    AOT torch.export: {type(e).__name__} -- cannot export, so `ir='dynamo'` is unavailable")
        print(f"        {str(e).splitlines()[0][:96]}")

    torch._dynamo.reset()
    counters.clear()
    jit = torch.compile(model, backend="tensorrt", options={"min_block_size": 1})
    with torch.no_grad():
        output, reference = jit(data), model(data)
    print(f"    JIT torch.compile: ok, {counters['stats'].get('unique_graphs', 0)} frames compiled, "
          f"max |eager - TensorRT| = {(output - reference).abs().max().item():.1e}")
    print("    the branch stays in Python; each side is its own graph")
    return

@case_mark
def case_saveability() -> None:
    """Only the AOT result is a artifact you can ship.

    AOT hands back a `GraphModule`, which `torch_tensorrt.save` writes with the
    engine embedded (see `../SaveLoad/`). JIT hands back a `torch.compile`
    wrapper around the original `nn.Module`; there is no compiled artifact to
    serialise, and the engines live in dynamo's cache for this process only.
    """
    model = Simple().cuda().eval()
    aot = compile_aot(model)
    torch._dynamo.reset()
    jit = torch.compile(model, backend="tensorrt", options={"min_block_size": 1})
    with torch.no_grad():
        jit(data)  # Force the build, so the comparison is fair

    for tag, module in [("AOT", aot), ("JIT", jit)]:
        save_file = work_path / f"{tag}.ep"
        try:
            torch_tensorrt.save(module, str(save_file), output_format="exported_program", arg_inputs=[data])
            print(f"    {tag}: saved {save_file.stat().st_size / 1024:.0f} KB")
        except Exception as e:
            print(f"    {tag}: cannot save -- {type(e).__name__}: {str(e).splitlines()[0][:80]}")
    print("    to ship a JIT-compiled model, recompile AOT or use engine caching (see ../EngineCaching/)")
    return

@case_mark
def case_backend_options() -> None:
    """Backend options are the same dictionary in both front ends.

    `torch_executed_ops` pins an operator to PyTorch whatever the converter says,
    which splits the graph around it. Useful to work around a converter bug or to
    keep an op in fp32 -- and a good way to see the partitioner at work.

    Accepted keys are the fields of `torch_tensorrt.dynamo.CompilationSettings`.
    """
    model = Simple().cuda().eval()
    for forced in [set(), {"torch.ops.aten.relu.default"}]:
        compiled = compile_aot(model, torch_executed_ops=forced)
        segment = [node.name for node in compiled.graph.nodes if node.op == "call_module"]
        print(f"    torch_executed_ops={str(forced) if forced else '{}':<36}: "
              f"{n_trt_engine(compiled)} TensorRT engine(s), segments {segment}")
    print("    `_run_on_gpu_*` is the piece left to PyTorch, sitting between two engines")
    return

if __name__ == "__main__":
    case_two_spellings()
    case_when_is_the_engine_built()
    case_data_dependent_control_flow()
    case_saveability()
    case_backend_options()

    print("\nFinish")
