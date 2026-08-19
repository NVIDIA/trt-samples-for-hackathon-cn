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
"""Replacing Torch-TensorRT's lowering of one operator with your own.

A converter turns one ATen node into TensorRT layers. Registering one for an
operator that Torch-TensorRT already handles is how you route around a converter
bug, or force a particular layer choice.

Two things decide whether your converter is the one that runs, and both fail
quietly when you get them wrong:

    priority              `STANDARD` **appends** to the list for that operator,
                          so a built-in converter still wins. Overriding needs
                          `HIGH`. See `case_standard_priority_does_nothing`.

    capability_validator  run per node before partitioning; returning False
                          hands the node to the next candidate, which is usually
                          the built-in one.

The example overrides `gelu` in `tanh` mode with an explicit polynomial, leaving
`erf` mode to the built-in converter.
"""

from typing import Dict, Sequence, Tuple, Union

import tensorrt as trt
import torch
import torch_tensorrt
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo import SourceIR
from torch_tensorrt.dynamo.conversion import ConversionContext, ConverterPriority, dynamo_tensorrt_converter, impl

from tensorrt_cookbook import case_mark

torch.manual_seed(31193)

# How many times each custom converter ran. A counter rather than a `print`, so
# the cases can assert on it instead of asking the reader to scan the log.
n_call = {"standard": 0, "high": 0}

class GeLU(torch.nn.Module):
    """`gelu` has two modes and TensorRT supports both; only `tanh` is overridden."""

    def __init__(self, mode: str = "tanh") -> None:
        """Remember the approximation mode."""
        super().__init__()
        self.mode = mode

    def forward(self, x):
        """Forward pass."""
        return torch.nn.functional.gelu(x, approximate=self.mode)

data = torch.randn(2, 5).cuda()
tanh_model = GeLU("tanh").cuda().eval()
erf_model = GeLU("none").cuda().eval()

def build_gelu_converter(tag: str):
    """A `gelu` converter that counts its own calls, tagged so priority is visible."""

    def convert(ctx: ConversionContext, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        """`gelu(Tensor self, *, str approximate='none') -> Tensor`, tanh approximation."""
        n_call[tag] += 1
        counter = [0]

        def unique() -> str:
            """TensorRT layer names have to be unique."""
            counter[0] += 1
            return f"{tag}_{counter[0]}"

        # `impl.*` are Torch-TensorRT's helpers over the raw TensorRT API; a
        # converter may also add `ctx.net` layers directly.
        mul = lambda x, y: impl.elementwise.mul(ctx, target, name=f"mul_{unique()}", source_ir=SourceIR.ATEN, lhs_val=x, rhs_val=y)
        add = lambda x, y: impl.elementwise.add(ctx, target, name=f"add_{unique()}", source_ir=SourceIR.ATEN, lhs_val=x, rhs_val=y)
        tanh = lambda x: impl.activation.tanh(ctx, target, name=f"tanh_{unique()}", source_ir=SourceIR.ATEN, input_val=x)

        # 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = args[0]
        inner = mul(mul(x, 0.79788456080000003), add(mul(mul(x, 0.044714999999999998), x), 1.0))
        return mul(mul(x, 0.5), add(tanh(inner), 1.0))

    return convert

def register(tag: str, priority: ConverterPriority) -> None:
    """Register the tagged converter for `gelu`, only for the tanh approximation."""
    dynamo_tensorrt_converter(
        torch.ops.aten.gelu.default,
        capability_validator=lambda node, settings: node.kwargs.get("approximate") == "tanh",
        supports_dynamic_shapes=True,
        priority=priority,
    )(build_gelu_converter(tag))
    return

def compile_and_report(tag: str, model, expected_caller: str) -> None:
    """Compile, then report which converter ran and how far the result drifted."""
    for key in n_call:
        n_call[key] = 0
    compiled = torch_tensorrt.compile(model, arg_inputs=(data, ), min_block_size=1)
    with torch.no_grad():
        reference, output = model(data), compiled(data)
    difference = (output - reference).abs().max().item()
    print(f"    {tag:<34}: calls {n_call}, max |eager - TensorRT| = {difference:.2e}")
    assert difference < 1e-5, "the custom converter changed the result"
    if expected_caller == "none":
        assert sum(n_call.values()) == 0, "a custom converter ran when it should not have"
    else:
        assert n_call[expected_caller] > 0, f"the {expected_caller} converter did not run"
    return

@case_mark
def case_builtin_baseline() -> None:
    """No custom converter registered yet: the built-in one handles `gelu`."""
    compile_and_report("built-in converter", tanh_model, "none")
    return

@case_mark
def case_standard_priority_does_nothing() -> None:
    """The trap: registering at `STANDARD` does not override anything.

    `STANDARD` **appends** to the candidate list for the target operator. The
    built-in `gelu` converter is already in that list and its validator passes,
    so it is chosen first and the new converter is never consulted.

    Nothing warns. The registration succeeds, the compile succeeds, the results
    are correct -- and the custom code never ran. `calls {'standard': 0}` below
    is the only evidence.
    """
    register("standard", ConverterPriority.STANDARD)
    compile_and_report("registered at STANDARD", tanh_model, "none")
    print("    zero calls: the built-in converter was still used")
    return

@case_mark
def case_high_priority_overrides() -> None:
    """`HIGH` prepends, so the custom converter is consulted first and wins.

    The result still matches eager PyTorch, which is the point of checking
    against eager rather than against the built-in TensorRT path: a converter
    that merely reproduces TensorRT's own answer proves much less.
    """
    register("high", ConverterPriority.HIGH)
    compile_and_report("registered at HIGH", tanh_model, "high")
    print("    one call: the polynomial above became the TensorRT layers")
    return

@case_mark
def case_validator_gates_by_node() -> None:
    """`capability_validator` decides per node, so `erf` mode falls through.

    Both custom converters are still registered at this point. Their validator
    asks for `approximate == "tanh"`, this module uses the default `erf`
    approximation, so neither fires and the built-in converter handles it.

    This is how a converter that only covers part of an operator's schema is
    written: claim the cases you handle and let the rest fall through.
    """
    compile_and_report("erf mode, validator rejects", erf_model, "none")
    print("    the node went to the built-in converter, no fallback to PyTorch was needed")
    return

if __name__ == "__main__":
    case_builtin_baseline()
    case_standard_priority_does_nothing()
    case_high_priority_overrides()
    case_validator_gates_by_node()

    print("\nFinish")
