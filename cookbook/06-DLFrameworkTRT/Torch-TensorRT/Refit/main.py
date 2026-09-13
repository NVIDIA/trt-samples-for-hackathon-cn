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
"""Swap new weights into a compiled module without recompiling: `refit_module_weights`.

Compilation is the expensive step. When only the *weights* change -- a new checkpoint, a
different LoRA adapter, a fine-tuned head -- rebuilding the engine wastes all of it, because
the graph is identical and only the numbers inside it moved.

`torch_tensorrt.dynamo.refit_module_weights(compiled, new_exported_program)` replaces the
weights in place. The rules that make it work, each measured below:

+ **The engine must be built for it.** `immutable_weights=False` at compile time. The default
  is an immutable engine, and refitting one is refused.
+ **The graph must be identical -- and nothing checks that for you.** Refit replaces weights,
  not structure, but feeding it a model with *more* layers is **accepted silently** and the
  extra layers are discarded. See `case_different_graph_is_accepted_and_wrong`.
+ **It is much faster than recompiling** -- that is the entire point, and the number is worth
  knowing for a deployment that swaps adapters at run time.

`04-Feature/Refit` covers the same feature at the TensorRT API level, where the weights are
named individually. Here the unit is a whole `ExportedProgram`, which is the natural shape
when the source of truth is a PyTorch module.
"""

import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch_tensorrt

from tensorrt_cookbook import case_mark

np.random.seed(31193)
torch.manual_seed(31193)

N_BATCH = 16
N_FEATURE = 512
N_LAYER = 6
N_WARMUP = 5
N_INFERENCE = 20

result = OrderedDict()

class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer_list = nn.ModuleList([nn.Linear(N_FEATURE, N_FEATURE) for _ in range(N_LAYER)])

    def forward(self, x):
        for layer in self.layer_list:
            x = torch.relu(layer(x))
        return x

def make_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return Model().eval().cuda()

def compile_module(model, example, *, b_refittable: bool):
    exported = torch.export.export(model, (example, ))
    return torch_tensorrt.dynamo.compile(
        exported,
        inputs=[example],
        use_explicit_typing=True,
        min_block_size=1,
        immutable_weights=not b_refittable,  # The flag that decides whether refit is possible
    )

# ================================================================ Cases

@case_mark
def case_compile_refittable() -> None:
    """Compile once, with refit enabled, and record what it cost."""
    model = make_model(31193)
    example = torch.randn(N_BATCH, N_FEATURE).cuda()

    t0 = time.time()
    compiled = compile_module(model, example, b_refittable=True)
    compile_second = time.time() - t0

    with torch.no_grad():
        reference = model(example)
    output = compiled(example)
    print(f"    compiled in {compile_second:.1f} s (immutable_weights=False)")
    print(f"    matches eager Torch: max |diff| = {float((output - reference).abs().max()):.3e}")

    result["example"] = example
    result["compiled"] = compiled
    result["compile_second"] = compile_second
    result["model_a"] = model
    return

@case_mark
def case_refit_with_new_weights() -> None:
    """Replace the weights with a different checkpoint and check the output follows."""
    example = result["example"]
    before = result["compiled"](example).detach().clone()

    model_b = make_model(999)  # A genuinely different set of weights
    with torch.no_grad():
        expected = model_b(example)
    exported_b = torch.export.export(model_b, (example, ))

    t0 = time.time()
    refitted = torch_tensorrt.dynamo.refit_module_weights(result["compiled"], exported_b)
    refit_second = time.time() - t0

    after = refitted(example)
    changed = float((after - before).abs().max())
    matches = float((after - expected).abs().max())
    print(f"    refitted in {refit_second:.2f} s")
    print(f"    output changed from the old weights : max |diff| = {changed:.3e}")
    print(f"    output matches the NEW eager model  : max |diff| = {matches:.3e}")
    assert changed > 1e-3, "The refit did not take effect"
    assert matches < 1e-2, "The refitted engine does not agree with the new weights"

    result["refit_second"] = refit_second
    result["refitted"] = refitted
    return

@case_mark
def case_refit_vs_recompile() -> None:
    """The number that justifies the feature."""
    example = result["example"]
    model_c = make_model(4242)

    t0 = time.time()
    compile_module(model_c, example, b_refittable=True)
    recompile_second = time.time() - t0

    refit_second = result["refit_second"]
    print(f"    full recompile : {recompile_second:7.2f} s")
    print(f"    refit          : {refit_second:7.2f} s")
    print(f"    speed-up       : {recompile_second / max(refit_second, 1e-6):7.1f}x")
    result["recompile_second"] = recompile_second
    return

@case_mark
def case_immutable_engine_is_refused() -> None:
    """An engine compiled the default way cannot be refitted, and says so."""
    example = result["example"]
    model = make_model(31193)
    immutable = compile_module(model, example, b_refittable=False)

    model_b = make_model(999)
    exported_b = torch.export.export(model_b, (example, ))
    try:
        torch_tensorrt.dynamo.refit_module_weights(immutable, exported_b)
        print("    refitting an immutable engine: ACCEPTED (unexpected)")
        result["immutable_refused"] = False
    except Exception as exception:  # noqa: BLE001 - the message is the lesson
        print(f"    refitting an immutable engine -> {type(exception).__name__}")
        print(f"        {str(exception).splitlines()[0][:130]}")
        result["immutable_refused"] = True
    return

@case_mark
def case_different_graph_is_accepted_and_wrong() -> None:
    """Refit with a **structurally different** model. This is the dangerous one.

    Refit replaces weights, not structure -- so a model with more layers should not be a
    valid source. Measured behaviour on `torch_tensorrt` 2.14.0a0:

        refitting a 6-layer engine from an 8-layer model  ->  ACCEPTED, no error
        result vs the true 8-layer output                 ->  7.8e-02  (wrong)
        result vs that model's FIRST SIX layers           ->  0.0      (exactly)

    **The extra two layers are silently discarded.** The engine keeps its own structure and
    takes as many weights as it has slots for. There is no exception, no warning, and the
    output is plausible -- it is a real model's output, just not the model you asked for.

    So the graph-identity precondition is **yours to enforce**. If a service refits from
    checkpoints it did not build, it must check the architecture itself.
    """
    example = result["example"]

    class Wider(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.layer_list = nn.ModuleList([nn.Linear(N_FEATURE, N_FEATURE) for _ in range(N_LAYER + 2)])

        def forward(self, x):
            for layer in self.layer_list:
                x = torch.relu(layer(x))
            return x

    torch.manual_seed(7)
    wider = Wider().eval().cuda()
    with torch.no_grad():
        wider_expected = wider(example)

    # The same weights, truncated to the engine's actual depth
    truncated = Model().eval().cuda()
    truncated.load_state_dict({f"layer_list.{i}.{k}": wider.state_dict()[f"layer_list.{i}.{k}"] for i in range(N_LAYER) for k in ("weight", "bias")})
    with torch.no_grad():
        truncated_expected = truncated(example)

    try:
        refitted = torch_tensorrt.dynamo.refit_module_weights(result["compiled"], torch.export.export(wider, (example, )))
        output = refitted(example)
        against_wider = float((output - wider_expected).abs().max())
        against_truncated = float((output - truncated_expected).abs().max())
        print(f"    refitting a {N_LAYER}-layer engine from a {N_LAYER + 2}-layer model: ACCEPTED, no error")
        print(f"        vs the true {N_LAYER + 2}-layer output : max |diff| = {against_wider:.3e}   <- WRONG")
        print(f"        vs that model's first {N_LAYER} layers : max |diff| = {against_truncated:.3e}   <- exactly this")
        print(f"    -> the extra {2} layers were silently discarded. Graph identity is YOUR precondition.")
        result["graph_refused"] = False
        assert against_truncated < 1e-4 < against_wider, "Expected the truncated-weights explanation"
    except Exception as exception:  # noqa: BLE001
        print(f"    refitting with a different graph -> {type(exception).__name__}: {str(exception).splitlines()[0][:110]}")
        result["graph_refused"] = True
    return

@case_mark
def case_summary() -> None:
    print("\n" + "    " + "=" * 66)
    print(f"    {'operation':<34}{'seconds':>12}{'vs recompile':>18}")
    print("    " + "-" * 66)
    print(f"    {'first compile (refittable)':<34}{result['compile_second']:>12.2f}{'-':>18}")
    print(f"    {'full recompile, new weights':<34}{result['recompile_second']:>12.2f}{1.0:>17.1f}x")
    print(f"    {'refit_module_weights':<34}{result['refit_second']:>12.2f}{result['recompile_second'] / max(result['refit_second'], 1e-6):>17.1f}x")
    print("    " + "=" * 66)
    print(f"    immutable engine refused : {result['immutable_refused']}")
    print(f"    different graph refused  : {result['graph_refused']}" + ("" if result["graph_refused"] else "   <- NOT refused: extra layers silently dropped"))
    return

def main() -> None:
    case_compile_refittable()
    case_refit_with_new_weights()
    case_refit_vs_recompile()
    case_immutable_engine_is_refused()
    case_different_graph_is_accepted_and_wrong()
    case_summary()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
