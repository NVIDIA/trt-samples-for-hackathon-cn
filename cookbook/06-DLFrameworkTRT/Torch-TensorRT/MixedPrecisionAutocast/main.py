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
"""Getting FP16 / BF16 out of a strongly typed TensorRT 11 network.

TensorRT 11 networks are **strongly typed**: the network says what precision each
operation runs at, and the builder does not go looking for a faster type on its
own. `enabled_precisions={torch.float16}`, which used to be the way to ask for
half precision, therefore does nothing -- and says nothing, see
`case_enabled_precisions_is_ignored`.

The replacement is Torch-TensorRT **Autocast**: `enable_autocast=True` plus
`autocast_low_precision_type=`, which rewrites the graph to the low type and
inserts casts, with per-node and per-op escape hatches for the parts that must
stay in FP32.

See also the cookbook skill `trt-strong-typing-migration` for the same change at
the `INetworkDefinition` and `trtexec` level.
"""

import time

import torch
import torch.nn as nn
import torch_tensorrt

from tensorrt_cookbook import case_mark

torch.manual_seed(31193)

N_FEATURE, N_DEPTH, BATCH = 1024, 8, 64
N_WARMUP, N_TEST = 30, 200

class Net(nn.Module):
    """A stack of square `Linear` layers, big enough for the type to matter."""

    def __init__(self, n_feature: int = N_FEATURE, n_depth: int = N_DEPTH) -> None:
        """Build the stack."""
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(n_feature, n_feature) for _ in range(n_depth)])

    def forward(self, x):
        """Forward pass."""
        for layer in self.layer:
            x = torch.relu(layer(x))
        return x

class PartiallyAutocastModel(nn.Module):
    """Half the graph sits inside a `torch.autocast` block, half does not."""

    def __init__(self) -> None:
        """Build the two halves."""
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8 * 32 * 32, 10)

    def forward(self, x):
        """Forward pass, with the second half under PyTorch autocast."""
        feature = torch.relu(self.conv(x)).flatten(1)
        with torch.autocast(x.device.type, enabled=True, dtype=torch.float16):
            logit = self.fc(feature)
        return feature, logit

model = Net().cuda().eval()
data = (torch.randn(BATCH, N_FEATURE).cuda(), )
exported = torch.export.export(model, data)
with torch.no_grad():
    reference = model(*data)
reference_scale = reference.abs().max().item()

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

def report(tag: str, module) -> None:
    """Latency, output dtype and relative error against eager FP32."""
    with torch.no_grad():
        output = module(*data)
    error = (output.to(torch.float32) - reference).abs().max().item() / reference_scale
    print(f"      {tag:<38}: {benchmark(module):6.3f} ms, out {str(output.dtype)[6:]:<9} max rel err {error:.2e}")
    return

def compile_it(**kwargs):
    """Compile the shared model with the given options."""
    return torch_tensorrt.compile(exported.module(), arg_inputs=data, min_block_size=1, **kwargs)

@case_mark
def case_autocast() -> None:
    """The baseline and the two low-precision types, measured.

    BF16 has the same exponent range as FP32 but three fewer mantissa bits than
    FP16, so it is the less accurate of the two here and no faster -- it earns
    its keep on models that overflow in FP16, which this one does not.
    """
    report("no autocast (all FP32)", compile_it())
    for low_type in [torch.float16, torch.bfloat16]:
        report(f"autocast, low type = {str(low_type)[6:]}", compile_it(enable_autocast=True, autocast_low_precision_type=low_type))
    return

@case_mark
def case_exclusions() -> None:
    """Keeping part of the graph in FP32.

    `autocast_excluded_ops` takes ATen op names and `autocast_excluded_nodes`
    takes regular expressions matched against node names, so a single layer can
    be pinned by name. Both are how a numerically sensitive block (a softmax, a
    normalisation, an accumulation) is kept out of the low type.

    Excluding the only expensive operator here removes the entire benefit --
    a useful reminder that the exclusion list is not free.

    The node-name route is shown but does **not** fire below: the pattern is
    matched against the lowered graph, whose names differ from the exported
    ones, and a pattern that matches nothing is silently ignored. That failure
    mode is the reason to prefer `autocast_excluded_ops`.
    """
    report("autocast FP16", compile_it(enable_autocast=True, autocast_low_precision_type=torch.float16))
    report("+ excluded op: aten.linear", compile_it(enable_autocast=True, autocast_low_precision_type=torch.float16, autocast_excluded_ops={"torch.ops.aten.linear.default"}))
    report("+ excluded nodes: ^linear$, ^linear_1$", compile_it(enable_autocast=True, autocast_low_precision_type=torch.float16, autocast_excluded_nodes={"^linear$", "^linear_1$"}))
    print("      row 2: excluding the only expensive operator removes the whole benefit -- FP16 is now")
    print("             slower than plain FP32, because all that is left in FP16 is casts and `relu`")
    print("      row 3: identical to row 1, i.e. this pattern matched nothing. `autocast_excluded_nodes`")
    print("             is matched against the node names of the *lowered* graph, not the ones")
    print("             `torch.export` shows, and a pattern that matches nothing fails silently.")
    print("             Prefer `autocast_excluded_ops` unless the exact lowered names are known.")
    return

@case_mark
def case_enabled_precisions_is_ignored() -> None:
    """The trap: the old way of asking for FP16 is silently a no-op.

    On a weakly typed network (TensorRT 10 and earlier) `enabled_precisions`
    listed the types the builder was *allowed* to pick from. TensorRT 11 networks
    are strongly typed, so there is nothing to pick: the graph already says FP32
    everywhere and the builder honours it.

    The call still succeeds, emits **no warning**, and returns an FP32 module.
    A model that was "converted to FP16" this way was never converted at all.
    """
    report("enabled_precisions={torch.float16}", compile_it(enabled_precisions={torch.float16}))
    report("enable_autocast=True  (the fix)", compile_it(enable_autocast=True, autocast_low_precision_type=torch.float16))
    print("      identical latency and zero error on the first row -- nothing happened, and nothing warned")
    return

@case_mark
def case_with_pytorch_autocast() -> None:
    """Torch-TensorRT Autocast composes with PyTorch's own `torch.autocast`.

    The `torch.autocast` block in the model already marks part of the graph FP16
    at export time; Torch-TensorRT Autocast then handles the rest. The two are
    independent, which is why the outputs can end up with different dtypes.
    """
    mixed = PartiallyAutocastModel().cuda().eval()
    image = (torch.randn(8, 3, 32, 32).cuda(), )
    mixed_exported = torch.export.export(mixed, image)
    with torch.no_grad():
        eager_feature, eager_logit = mixed(*image)

    compiled = torch_tensorrt.compile(mixed_exported.module(), arg_inputs=image, min_block_size=1, enable_autocast=True, autocast_low_precision_type=torch.bfloat16)
    with torch.no_grad():
        feature, logit = compiled(*image)

    print(f"      feature (outside torch.autocast): eager {str(eager_feature.dtype)[6:]:<9} -> TensorRT {str(feature.dtype)[6:]}")
    print(f"      logit   (inside  torch.autocast): eager {str(eager_logit.dtype)[6:]:<9} -> TensorRT {str(logit.dtype)[6:]}")
    for tag, got, want in [("feature", feature, eager_feature), ("logit", logit, eager_logit)]:
        error = (got.to(torch.float32) - want.to(torch.float32)).abs().max().item() / want.abs().max().item()
        print(f"      {tag:<8} max rel err {error:.2e}")
    return

if __name__ == "__main__":
    case_autocast()
    case_exclusions()
    case_enabled_precisions_is_ignored()
    case_with_pytorch_autocast()

    print("\nFinish")
