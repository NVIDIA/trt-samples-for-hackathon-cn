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
"""Weight streaming: run a model whose weights do not fit, by keeping most of them in host memory.

An engine normally holds all of its weights in device memory. Weight streaming lifts that
requirement: the weights live in host memory and are copied in as each layer needs them, so
the device footprint becomes a **budget you choose** rather than a property of the model.
The cost is bandwidth -- every inference re-fetches whatever did not stay resident.

Two APIs, and they are not interchangeable:

+ `enable_weight_streaming=True` at **compile** time. This is permission: the engine is built
  so that streaming is possible. Without it, the runtime knob below does nothing.
+ `torch_tensorrt.runtime.weight_streaming(module)` at **run** time. This is the context
  manager that exposes `device_budget`, the number of bytes of weights you are willing to
  keep resident.

The upstream example (`.../weight_streaming_example.py`) drives this with a gated Llama-2
download. This cookbook does not download at run time, so the model here is a deliberately
weight-heavy stack of large linear layers -- what matters for streaming is the ratio of
weights to activations, and a stack of big GEMMs has exactly the shape of a transformer's
feed-forward path.
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

N_BATCH = 8
N_FEATURE = 4096
N_LAYER = 12  # 12 x 4096 x 4096 x 2 byte ~= 400 MiB of weights
N_WARMUP = 3
N_INFERENCE = 10

result = OrderedDict()

class WeightHeavy(nn.Module):
    """A stack of large linear layers: many weights, few activations."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_list = nn.ModuleList([nn.Linear(N_FEATURE, N_FEATURE, bias=False) for _ in range(N_LAYER)])

    def forward(self, x):
        for layer in self.layer_list:
            x = torch.relu(layer(x))
        return x

def measure(module, input_tensor) -> float:
    """Median latency in ms."""
    for _ in range(N_WARMUP):
        module(input_tensor)
    torch.cuda.synchronize()
    latency_list = []
    for _ in range(N_INFERENCE):
        t0 = time.time()
        module(input_tensor)
        torch.cuda.synchronize()
        latency_list.append((time.time() - t0) * 1000)
    return float(np.median(latency_list))

# ================================================================ Cases

@case_mark
def case_compile() -> None:
    """Compile twice: once ordinarily, once with weight streaming permitted."""
    model = WeightHeavy().half().eval().cuda()
    n_weight_byte = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"    model: {N_LAYER} x Linear({N_FEATURE}, {N_FEATURE}) fp16 = {n_weight_byte / (1 << 20):.0f} MiB of weights")

    example = torch.randn(N_BATCH, N_FEATURE, dtype=torch.half, device="cuda")
    exported = torch.export.export(model, (example, ))

    compiled_plain = torch_tensorrt.dynamo.compile(exported, inputs=[example], use_explicit_typing=True, min_block_size=1)
    compiled_stream = torch_tensorrt.dynamo.compile(exported, inputs=[example], use_explicit_typing=True, min_block_size=1, enable_weight_streaming=True)

    result["example"] = example
    result["plain"] = compiled_plain
    result["stream"] = compiled_stream
    result["n_weight_byte"] = n_weight_byte
    result["reference"] = model(example).detach()
    print("    compiled both: enable_weight_streaming=False and True")
    return

@case_mark
def case_budget_range() -> None:
    """What budgets the engine will accept, and what they mean.

    `device_budget` is settable inside the context manager. Reading it back tells you the
    automatic budget; the maximum is the whole weight set, and the minimum is what the engine
    needs to run at all.
    """
    with torch_tensorrt.runtime.weight_streaming(result["stream"]) as streaming:
        automatic = streaming.device_budget
        total = streaming.total_device_budget
        print(f"    automatic budget : {automatic / (1 << 20):>9.1f} MiB")
        print(f"    total weight size: {total / (1 << 20):>9.1f} MiB")
        minimum = streaming.get_automatic_weight_streaming_budget()
        print(f"    automatic-mode budget from the API: {minimum / (1 << 20):>9.1f} MiB")
        result["budget"] = (automatic, total, minimum)
    return

@case_mark
def case_budget_sweep() -> None:
    """Latency against budget: the trade this feature exists to let you make."""
    _, total, _ = result["budget"]
    example = result["example"]

    plain_latency = measure(result["plain"], example)
    print(f"    no weight streaming            : {plain_latency:7.3f} ms")
    result["plain_latency"] = plain_latency

    row_list = []
    with torch_tensorrt.runtime.weight_streaming(result["stream"]) as streaming:
        for fraction in [1.0, 0.5, 0.25, 0.1, 0.0]:
            budget = int(total * fraction)
            streaming.device_budget = budget
            actual = streaming.device_budget
            latency = measure(result["stream"], example)
            output = result["stream"](example)
            difference = float((output.float() - result["reference"].float()).abs().max())
            print(f"    budget {fraction:>5.0%} = {actual / (1 << 20):>8.1f} MiB : {latency:7.3f} ms   max |diff| vs torch = {difference:.3e}")
            row_list.append((fraction, actual, latency, difference))
    result["sweep"] = row_list
    return

@case_mark
def case_summary() -> None:
    """What the budget bought, and what it cost."""
    _, total, _ = result["budget"]
    print("\n" + "    " + "=" * 74)
    print(f"    {'budget':>10}{'resident MiB':>16}{'latency ms':>14}{'vs no streaming':>18}")
    print("    " + "-" * 74)
    print(f"    {'(disabled)':>10}{result['n_weight_byte'] / (1 << 20):>16.1f}{result['plain_latency']:>14.3f}{1.0:>17.2f}x")
    for fraction, actual, latency, _ in result["sweep"]:
        print(f"    {fraction:>9.0%}{actual / (1 << 20):>16.1f}{latency:>14.3f}{latency / result['plain_latency']:>17.2f}x")
    print("    " + "=" * 74)
    print("    A budget of 0 keeps no weights resident: every inference re-fetches all of them,")
    print("    which is the configuration that lets a model larger than VRAM run at all.")
    return

def main() -> None:
    case_compile()
    case_budget_range()
    case_budget_sweep()
    case_summary()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
