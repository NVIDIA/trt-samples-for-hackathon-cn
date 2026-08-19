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
"""Persisting a compiled module, and keeping its dynamic shapes alive across the trip.

`torch_tensorrt.save` / `torch_tensorrt.load` write the built engine into an
`ExportedProgram`, so loading is a deserialization rather than a rebuild.

The catch is that a dynamic model does not stay dynamic by itself. `save` defaults
to `retrace=True`, which re-exports the module -- and a re-export with no shape
spec specializes on whatever `arg_inputs` happens to be, silently turning the
model static. `case_what_preserves_dynamism` maps out exactly when that happens.

See `../DynamicShapes/` for how the dynamic dimension is declared in the first
place; this example is only about surviving save and load.
"""

import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch_tensorrt

from tensorrt_cookbook import case_mark

torch.manual_seed(31193)

N_FEATURE_IN, N_FEATURE_OUT = 10, 5
BATCH_MIN, BATCH_OPT, BATCH_MAX = 1, 8, 32
BATCH_EXAMPLE = 4  # Deliberately not `opt`, so a wrong specialization is obvious

work_path = Path(tempfile.mkdtemp(prefix="torch-trt-saveload-"))

class SimpleModel(nn.Module):
    """One `Linear`, enough to own an engine and nothing more."""

    def __init__(self) -> None:
        """Build the layer."""
        super().__init__()
        self.linear = nn.Linear(N_FEATURE_IN, N_FEATURE_OUT)

    def forward(self, x):
        """Forward pass."""
        return self.linear(x)

class ConvModel(nn.Module):
    """A conv, so height and width can be dynamic too."""

    def __init__(self) -> None:
        """Build the layer."""
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, x):
        """Forward pass."""
        return self.conv(x)

model = SimpleModel().eval().cuda()
example = torch.randn(BATCH_EXAMPLE, N_FEATURE_IN).cuda()
dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=BATCH_MIN, max=BATCH_MAX)}}
input_spec = [torch_tensorrt.Input(min_shape=(BATCH_MIN, N_FEATURE_IN), opt_shape=(BATCH_OPT, N_FEATURE_IN), max_shape=(BATCH_MAX, N_FEATURE_IN), dtype=torch.float32, name="x")]

def compile_dynamic():
    """Compile `model` with a dynamic batch dimension."""
    exported = torch.export.export(model, (example, ), dynamic_shapes=dynamic_shapes, strict=False)
    return torch_tensorrt.dynamo.compile(exported, inputs=input_spec, min_block_size=1)

def usable_batch(module, batch_list: list) -> list:
    """Which of `batch_list` the module actually accepts."""
    result = []
    for batch in batch_list:
        try:
            with torch.no_grad():
                module(torch.randn(batch, N_FEATURE_IN).cuda())
            result.append(str(batch))
        except Exception:
            result.append(f"{batch}(x)")
    return result

@case_mark
def case_roundtrip() -> None:
    """Save, load, and check that the loaded module is the same engine.

    Two things worth checking and neither is obvious from the API: the outputs
    have to match the pre-save module **bit for bit** (a rebuilt engine would
    likely differ in the last ULP), and loading has to be fast (a rebuild would
    take seconds).
    """
    compiled = compile_dynamic()
    save_file = work_path / "roundtrip.ep"

    torch_tensorrt.save(compiled, str(save_file), output_format="exported_program", arg_inputs=[example], dynamic_shapes=dynamic_shapes)

    t0 = time.time()
    loaded = torch_tensorrt.load(str(save_file)).module()
    load_second = time.time() - t0

    print(f"    file {save_file.stat().st_size / 1024:.0f} KB, loaded in {load_second:.3f} s")
    for batch in [BATCH_MIN, BATCH_EXAMPLE, BATCH_OPT, BATCH_MAX]:
        data = torch.randn(batch, N_FEATURE_IN).cuda()
        with torch.no_grad():
            before, after = compiled(data), loaded(data)
        difference = (before - after).abs().max().item()
        print(f"      batch={batch:<3}: max |before save - after load| = {difference:.1e}")
        assert difference == 0.0, "the loaded module is not the same engine"
    print("    bit-identical and loaded in milliseconds, so the engine was deserialized, not rebuilt")
    return

@case_mark
def case_what_preserves_dynamism() -> None:
    """The trap: `save` re-exports by default, and a re-export can specialize.

    `retrace` defaults to **True**, `dynamic_shapes` defaults to **None**. So the
    obvious call --

        torch_tensorrt.save(compiled, path, arg_inputs=[example])

    -- re-exports against a plain example tensor, which pins the batch dimension
    to that tensor's size. Nothing warns. The model still loads and still returns
    correct answers *for that one batch size*, and fails a guard for every other.

    Two ways out, both shown below: hand `save` the `dynamic_shapes` dict, or
    hand it `torch_tensorrt.Input(min/opt/max)` objects instead of tensors and
    let it infer. Turning `retrace` off also works, because then nothing is
    re-exported.
    """
    batch_list = [BATCH_MIN, BATCH_EXAMPLE, BATCH_OPT, BATCH_MAX]
    compiled = compile_dynamic()
    print(f"    before saving      : batches ok {usable_batch(compiled, batch_list)}")

    variant = [
        ("default", "tensor, no spec, retrace=True ", dict(arg_inputs=[example]), "<- the default call, and it is wrong"),
        ("spec", "tensor, dynamic_shapes=       ", dict(arg_inputs=[example], dynamic_shapes=dynamic_shapes), ""),
        ("noretrace", "tensor, retrace=False         ", dict(arg_inputs=[example], retrace=False), ""),
        ("inferred", "Input(min/opt/max), inferred  ", dict(arg_inputs=input_spec), ""),
    ]
    for name, tag, option, note in variant:
        save_file = work_path / f"variant-{name}.ep"
        torch_tensorrt.save(compiled, str(save_file), output_format="exported_program", **option)
        loaded = torch_tensorrt.load(str(save_file)).module()
        print(f"      {tag}: batches ok {usable_batch(loaded, batch_list)}  {note}")
    print(f"    `(x)` means the guard rejected it, e.g. `Guard failed: x.size()[0] == {BATCH_EXAMPLE}`")
    return

@case_mark
def case_multiple_dynamic_dimension() -> None:
    """Batch, height and width all dynamic, through one `Input` spec.

    Inferring from `Input` scales to several dimensions without writing a
    `Dim` per axis, which is the main reason to prefer it over the explicit
    `dynamic_shapes` dict when the shapes are simple ranges.
    """
    conv_model = ConvModel().eval().cuda()
    spec = [torch_tensorrt.Input(min_shape=(1, 3, 64, 64), opt_shape=(8, 3, 256, 256), max_shape=(16, 3, 512, 512), dtype=torch.float32, name="image")]
    compiled = torch_tensorrt.compile(conv_model, ir="dynamo", inputs=spec, min_block_size=1)

    save_file = work_path / "multi_dim.ep"
    torch_tensorrt.save(compiled, str(save_file), output_format="exported_program", arg_inputs=spec)
    loaded = torch_tensorrt.load(str(save_file)).module()

    for shape in [(4, 3, 128, 128), (12, 3, 384, 384), (1, 3, 64, 64), (16, 3, 512, 512)]:
        data = torch.randn(*shape).cuda()
        with torch.no_grad():
            before, after = compiled(data), loaded(data)
        print(f"      {str(shape):<22}: output {tuple(after.shape)}, max |before - after| = {(before - after).abs().max().item():.1e}")
    return

if __name__ == "__main__":
    case_roundtrip()
    case_what_preserves_dynamism()
    case_multiple_dynamic_dimension()

    print("\nFinish")
