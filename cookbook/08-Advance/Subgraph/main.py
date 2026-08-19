# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
"""Parsing an ONNX file that contains sub-graphs into TensorRT.

`torch.jit.script` keeps a data-dependent `for` and an `if` as real control flow,
so the exported ONNX has a `Loop` whose body holds an `If` -- two levels of
`GraphProto` nested inside the main graph.

The interesting part is what TensorRT does with them: `INetworkDefinition` has no
notion of nesting, so both sub-graphs are **flattened into one layer list** and
the control flow survives only as boundary layer types. See `case_print_network`.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

from tensorrt_cookbook import (
    TRTWrapperV1,
    case_mark,
    export_network_as_onnx,
    print_network,
)

output_path = Path(__file__).parent
onnx_file = output_path / "example.onnx"
data = {"items": np.array([1, 2, 3, 4], dtype=np.float32)}

@torch.jit.script
def sum_even(items):
    """Sum the even elements. Scripted, so the `for` and the `if` stay in the graph."""
    s = torch.zeros(1, dtype=torch.float32)
    for c in items:
        if c % 2 == 0:
            s += c
    return s

class ExampleModel(torch.nn.Module):
    """Thin wrapper so the scripted function can be exported."""

    def forward(self, items):
        """Forward pass."""
        return sum_even(items)

def export_to_onnx() -> None:
    """Export the model, keeping the loop and the branch as ONNX control flow."""
    torch.onnx.export(
        ExampleModel().eval(),
        (torch.zeros(4, dtype=torch.float32), ),
        onnx_file,
        input_names=["items"],
        output_names=["sum_even"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={"items": {
            0: "N"
        }},
        dynamo=False,
    )
    return

def parse_onnx(tw: TRTWrapperV1) -> None:
    """Parse `example.onnx` into the wrapper's network and set the shape profile."""
    parser = trt.OnnxParser(tw.network, tw.logger)
    if not parser.parse_from_file(str(onnx_file)):
        raise RuntimeError(f"{onnx_file}: {parser.get_error(0)}")
    tw.profile.set_shape(tw.network.get_input(0).name, [1], [4], [64])
    return

@case_mark
def case_build_and_run() -> None:
    """Build and run, then check the result against PyTorch."""
    tw = TRTWrapperV1(logger=trt.Logger.Severity.ERROR)
    parse_onnx(tw)
    tw.build()
    tw.setup(data)
    tw.infer()

    reference = ExampleModel().eval()(torch.from_numpy(data["items"])).detach().numpy()
    output = tw.buffer["sum_even"][0]
    print(f"    PyTorch  : {reference}")
    print(f"    TensorRT : {output}")
    print(f"    AllClose : {np.allclose(output, reference, atol=1e-6)}")
    return

@case_mark
def case_print_network() -> None:
    """What the sub-graphs look like once TensorRT has them.

    ONNX nests: the main graph holds a `Loop`, whose body holds an `If`. TensorRT
    does not nest at all -- `print_network` walks one flat list, and the two
    sub-graphs show up only as boundary layers:

        Loop  ->  TRIP_LIMIT / RECURRENCE / ITERATOR / LOOP_OUTPUT
        If    ->  CONDITION / CONDITIONAL_INPUT / CONDITIONAL_OUTPUT

    Everything that was *inside* a body (`Gather`, `Div`, `Add`, ...) sits in the
    same list as the main-graph layers, with nothing marking which body it came
    from. That is why a network dump of a control-flow model reads oddly: the
    nesting is gone, only the boundary layers imply it.
    """
    tw = TRTWrapperV1(logger=trt.Logger.Severity.ERROR)
    parse_onnx(tw)
    network = tw.network

    kind = Counter(str(network.get_layer(i).type).split(".")[-1] for i in range(network.num_layers))
    print(f"    ONNX: 8 nodes in the main graph, 12 in the `Loop` body, 2 in the `If` branches")
    print(f"    TRT : {network.num_layers} layers, one flat list, no nesting")
    print(f"    layer types: {dict(kind)}")
    print(f"    loop boundary layers      : {sum(kind[k] for k in ['TRIP_LIMIT', 'RECURRENCE', 'ITERATOR', 'LOOP_OUTPUT'])}")
    print(f"    condition boundary layers : {sum(kind[k] for k in ['CONDITION', 'CONDITIONAL_INPUT', 'CONDITIONAL_OUTPUT'])}")

    # `layer.metadata` is the only trace of where a layer came from, and it is
    # not symmetric: the parser tags `If` branches with the body they belong to,
    # but says nothing at all for the `Loop` body. So a layer that came out of
    # the loop is indistinguishable from a main-graph layer.
    tagged = [i for i in range(network.num_layers) if "sub_graph" in network.get_layer(i).metadata]
    print(f"\n    layers whose `metadata` names the sub-graph they came from: {tagged}")
    for i in tagged[:2]:
        print(f"        {i:3d} {network.get_layer(i).metadata}")
    print(f"    ... all of them are `If` branches. The `Loop` body layers carry only their own")
    print(f"    ONNX node name, e.g. layer 19 -> {network.get_layer(19).metadata!r}, with no hint")
    print(f"    that it ran inside a loop. Provenance for `Loop` bodies is simply not recorded.")

    print(f"\n    Full dump from `print_network` follows, {network.num_layers} layers:\n")
    print_network(network)
    return

@case_mark
def case_export_network_as_onnx() -> None:
    """Round-trip the flattened network back out as an ONNX-like file for Netron.

    The result is *not* a valid ONNX model -- `TripLimit`, `Recurrence` and
    `ConditionalInput` are TensorRT layer types, not ONNX operators -- but it is
    the only way to see the parsed network as a picture, and it makes the point
    above visible: one flat graph, no `Loop` node to click into.
    """
    tw = TRTWrapperV1(logger=trt.Logger.Severity.ERROR)
    parse_onnx(tw)
    export_file = output_path / "network.onnx"
    export_network_as_onnx(tw.network, export_file)

    import onnx
    model = onnx.load(export_file)
    print(f"    exported {export_file.name}: {len(model.graph.node)} nodes, "
          f"{sum(1 for n in model.graph.node for a in n.attribute if a.type == onnx.AttributeProto.GRAPH)} of them hold a sub-graph")
    print(f"    node types: {dict(Counter(n.op_type for n in model.graph.node))}")
    return

if __name__ == "__main__":
    export_to_onnx()
    case_build_and_run()
    case_print_network()
    case_export_network_as_onnx()

    print("\nFinish")
