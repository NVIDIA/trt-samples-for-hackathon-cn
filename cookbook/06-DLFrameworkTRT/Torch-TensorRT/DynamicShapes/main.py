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
"""One engine that serves a range of input shapes, instead of one engine per shape.

Three ways to say "this dimension varies", and they are not interchangeable:

    JIT  `torch._dynamo.mark_dynamic(x, index, min, max)` + `torch.compile`
    AOT  `torch_tensorrt.Input(min_shape=, opt_shape=, max_shape=)`
    AOT  `torch.export.Dim` + `torch.export.export(dynamic_shapes=)`

The model is a Vision-Transformer-shaped block: a class token that must
`expand` to the batch size, a `cat`, a QKV projection and a `reshape` that
mentions the batch size again. Those are the operations that go wrong when a
batch size gets baked in, which `case_static_baseline` shows first.

The three declarations produce the same numbers, but they behave differently the
moment an input falls outside the declared range -- see `case_out_of_range`.
"""

import time

import torch
import torch.nn as nn
import torch_tensorrt

from tensorrt_cookbook import case_mark

torch.manual_seed(31193)

EMBED_DIM = 768
N_TOKEN = 196
N_HEAD = 12
BATCH_MIN, BATCH_OPT, BATCH_MAX = 1, 4, 32
N_WARMUP, N_TEST = 20, 100

class ExpandReshapeModel(nn.Module):
    """A ViT-style block whose shape handling depends on the batch size."""

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        """Build the class token and the QKV projection."""
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)

    def forward(self, x):
        """Prepend the class token, project to QKV, split the heads."""
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.qkv_proj(x)
        return x.reshape(batch_size, x.size(1), 3, N_HEAD, -1)

model = ExpandReshapeModel().cuda().eval()

def n_trt_submodule(module) -> int:
    """How many sub-graphs actually became TensorRT engines.

    Zero means the whole model silently stayed in eager PyTorch. Nothing raises
    when that happens, so every case below reports this number -- a latency
    comparison against a module that never reached TensorRT is meaningless, and
    that is exactly the trap `case_min_block_size` is about.
    """
    return sum(1 for node in module.graph.nodes if node.op == "call_module")

def data_of(batch: int) -> torch.Tensor:
    """One input tensor of the given batch size."""
    return torch.randn(batch, N_TOKEN, EMBED_DIM).cuda()

def check_range(module, batch_list: list, tag: str = "") -> None:
    """Run `module` at each batch size and compare against eager PyTorch."""
    for batch in batch_list:
        data = data_of(batch)
        with torch.no_grad():
            output, reference = module(data), model(data)
        difference = (output - reference).abs().max().item()
        print(f"      batch={batch:<3}{tag}: output {tuple(output.shape)}, max |eager - TensorRT| = {difference:.1e}")
    return

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

@case_mark
def case_static_baseline() -> None:
    """Why bother: a statically compiled engine only accepts the traced shape.

    Nothing here is dynamic, so the batch size is baked into the `expand` and
    the `reshape`. Feeding a different one does not merely run slowly, it fails.

    `min_block_size=1` is not decoration -- at the default of 5 this model is
    left in eager PyTorch entirely and the failure below would come from
    PyTorch's `cat` rather than from TensorRT. See `case_min_block_size`.
    """
    exported = torch.export.export(model, (data_of(BATCH_OPT), ))
    compiled = torch_tensorrt.dynamo.compile(exported, (data_of(BATCH_OPT), ), min_block_size=1)
    print(f"      TensorRT sub-modules: {n_trt_submodule(compiled)}")
    check_range(compiled, [BATCH_OPT])
    try:
        with torch.no_grad():
            compiled(data_of(BATCH_OPT * 2))
        print(f"      batch={BATCH_OPT * 2}  : unexpectedly accepted")
    except Exception as e:
        print(f"      batch={BATCH_OPT * 2}  : rejected, {type(e).__name__}: {str(e).splitlines()[0][:88]}")
    return

@case_mark
def case_aot_input_range() -> None:
    """AOT declaration 1: `torch_tensorrt.Input(min_shape / opt_shape / max_shape)`.

    The shortest way to get a shape range. `opt_shape` is what TensorRT tunes
    its tactics for, so it should be the shape actually seen most often -- the
    engine still runs at the other sizes, just not as well tuned.
    """
    compiled = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=torch_tensorrt.Input(
            min_shape=[BATCH_MIN, N_TOKEN, EMBED_DIM],
            opt_shape=[BATCH_OPT, N_TOKEN, EMBED_DIM],
            max_shape=[BATCH_MAX, N_TOKEN, EMBED_DIM],
            dtype=torch.float32,
        ),
        min_block_size=1,
    )
    print(f"      TensorRT sub-modules: {n_trt_submodule(compiled)}")
    check_range(compiled, [BATCH_MIN, BATCH_OPT, 17, BATCH_MAX])
    print("      one engine covers the whole range, no recompilation in between")
    return

@case_mark
def case_aot_export_dim() -> None:
    """AOT declaration 2: `torch.export.Dim` on an `ExportedProgram`.

    More verbose, but the constraint lives on the exported program, so
    `torch.export` itself validates the model against it (guards, `0/1`
    specialisation, dependent dimensions). Prefer this when the graph must be
    exported anyway, or when several dimensions are related.
    """
    batch = torch.export.Dim("batch", min=BATCH_MIN, max=BATCH_MAX)
    exported = torch.export.export(model, (data_of(BATCH_OPT), ), dynamic_shapes={"x": {0: batch}})
    compiled = torch_tensorrt.dynamo.compile(exported, (data_of(BATCH_OPT), ), min_block_size=1)
    print(f"      TensorRT sub-modules: {n_trt_submodule(compiled)}")
    check_range(compiled, [BATCH_MIN, BATCH_OPT, 17, BATCH_MAX])
    return

@case_mark
def case_jit_mark_dynamic() -> None:
    """JIT declaration: `torch._dynamo.mark_dynamic` + `torch.compile`.

    The bound is attached to the *tensor*, not to the compile call, and nothing
    is built until the first call. Note `mark_dynamic` mutates the tensor it is
    given, so it has to be applied to the tensor that triggers the compilation.
    """
    torch._dynamo.reset()
    data = data_of(BATCH_OPT)
    torch._dynamo.mark_dynamic(data, index=0, min=BATCH_MIN, max=BATCH_MAX)
    compiled = torch.compile(model, backend="tensorrt", options={"min_block_size": 1})

    t0 = time.time()
    with torch.no_grad():
        compiled(data)
    print(f"      first call builds the engine: {time.time() - t0:.2f} s")
    check_range(compiled, [BATCH_OPT, 17, BATCH_MAX])
    return

@case_mark
def case_out_of_range() -> None:
    """The one place the three declarations really differ.

    AOT hands TensorRT a profile with hard bounds, so an input outside it is a
    runtime error from `setInputShape`. JIT keeps dynamo in the loop, so an
    out-of-range input is just another guard failure: it recompiles, which costs
    seconds but does not fail.

    Neither is "right". AOT is predictable and fails loudly, which is what a
    deployed service usually wants; JIT keeps working at the cost of a stall.
    """
    batch = torch.export.Dim("batch", min=BATCH_MIN, max=BATCH_MAX)
    exported = torch.export.export(model, (data_of(BATCH_OPT), ), dynamic_shapes={"x": {0: batch}})
    aot = torch_tensorrt.dynamo.compile(exported, (data_of(BATCH_OPT), ), min_block_size=1)
    try:
        with torch.no_grad():
            aot(data_of(BATCH_MAX + 1))
        print(f"      AOT batch={BATCH_MAX + 1}: unexpectedly accepted")
    except Exception as e:
        print(f"      AOT batch={BATCH_MAX + 1}: {type(e).__name__} from the runtime, engine profile is a hard bound")

    torch._dynamo.reset()
    data = data_of(BATCH_OPT)
    torch._dynamo.mark_dynamic(data, index=0, min=BATCH_MIN, max=BATCH_MAX)
    jit = torch.compile(model, backend="tensorrt", options={"min_block_size": 1})
    with torch.no_grad():
        jit(data)  # Build for the declared range

    over = data_of(BATCH_MAX + 1)
    t0 = time.time()
    with torch.no_grad():
        output = jit(over)
    second = time.time() - t0
    print(f"      JIT batch={BATCH_MAX + 1}: accepted after {second:.2f} s -- dynamo re-traced and rebuilt")

    t0 = time.time()
    with torch.no_grad():
        jit(data_of(BATCH_MAX * 2))
    print(f"      JIT batch={BATCH_MAX * 2}: {time.time() - t0:.2f} s -- the rebuilt engine already covers it")
    print(f"      (numerics stay correct throughout: max diff {(output - model(over)).abs().max().item():.1e})")
    return

@case_mark
def case_min_block_size() -> None:
    """A trap found while writing `case_cost_of_being_dynamic`: silent fallback.

    `min_block_size` defaults to 5 -- a sub-graph with fewer operators is left in
    eager PyTorch rather than converted. This model lowers to three operators
    (`cat`, `linear`, `reshape`), so at the default **the static compile produces
    no engine at all** and nothing says so: no warning, no exception, and the
    module still returns correct results.

    The dynamic version clears the threshold only because the symbolic-shape
    arithmetic adds nodes. So at the default, "static vs dynamic" was really
    "eager vs TensorRT", and the first draft of this example measured the dynamic
    engine as *faster* than the static one. Always check that an engine exists
    before comparing anything.
    """
    static_exported = torch.export.export(model, (data_of(BATCH_OPT), ))
    batch = torch.export.Dim("batch", min=BATCH_MIN, max=BATCH_MAX)
    dynamic_exported = torch.export.export(model, (data_of(BATCH_OPT), ), dynamic_shapes={"x": {0: batch}})

    data = data_of(BATCH_OPT)
    for min_block_size in [5, 1]:
        static = torch_tensorrt.dynamo.compile(static_exported, (data_of(BATCH_OPT), ), min_block_size=min_block_size)
        dynamic = torch_tensorrt.dynamo.compile(dynamic_exported, (data_of(BATCH_OPT), ), min_block_size=min_block_size)
        note = "  <- the default, and the static model never reaches TensorRT" if min_block_size == 5 else ""
        print(f"      min_block_size={min_block_size}: static engines={n_trt_submodule(static)} {benchmark(static, data):.3f} ms | "
              f"dynamic engines={n_trt_submodule(dynamic)} {benchmark(dynamic, data):.3f} ms{note}")
    return

@case_mark
def case_cost_of_being_dynamic() -> None:
    """What the shape range costs at the shape a static engine would have used.

    Measured answer: essentially nothing at `opt_shape`, which is what TensorRT
    tuned its tactics for. Moving away from `opt_shape` is where a dynamic engine
    gives something up -- and even that is small here.
    """
    static_exported = torch.export.export(model, (data_of(BATCH_OPT), ))
    static = torch_tensorrt.dynamo.compile(static_exported, (data_of(BATCH_OPT), ), min_block_size=1)

    batch = torch.export.Dim("batch", min=BATCH_MIN, max=BATCH_MAX)
    dynamic_exported = torch.export.export(model, (data_of(BATCH_OPT), ), dynamic_shapes={"x": {0: batch}})
    dynamic = torch_tensorrt.dynamo.compile(dynamic_exported, (data_of(BATCH_OPT), ), min_block_size=1)

    data = data_of(BATCH_OPT)
    static_ms, dynamic_ms = benchmark(static, data), benchmark(dynamic, data)
    print(f"      at batch={BATCH_OPT} (= opt_shape): static {static_ms:.3f} ms, dynamic {dynamic_ms:.3f} ms "
          f"({dynamic_ms / static_ms:.2f}x)")
    for batch_size in [BATCH_MIN, BATCH_MAX]:
        print(f"      at batch={batch_size:<3}          : dynamic {benchmark(dynamic, data_of(batch_size)):.3f} ms")
    return

if __name__ == "__main__":
    case_static_baseline()
    case_aot_input_range()
    case_aot_export_dim()
    case_jit_mark_dynamic()
    case_out_of_range()
    case_min_block_size()
    case_cost_of_being_dynamic()

    print("\nFinish")
