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
"""P1 -- the read-only projection the mining algorithm works on.

Deliberately independent of both `onnx` and `onnx_graphsurgeon`: the discovery
(P2), verification (P3) and selection (P4) stages only ever see `GraphIR`, so
swapping the graph library later touches the import and the rewrite stages only.

Three modelling decisions, each one load-bearing:

1. initializers are NOT nodes. Two instances of the same block always hold
   different weights, so a graph where weights are nodes has no repetition at
   all. Weight dtype/shape is recorded on the consuming node instead.
2. edges carry `(out_slot, in_slot)`. ONNX operator inputs are ordered
   (`Sub(a, b) != Sub(b, a)`) and multi-output operators such as `Split` make
   the producing slot matter too.
3. the node label is tunable, see `STRICTNESS_LIST` in config.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import onnx_graphsurgeon as gs

@dataclass(frozen=True)
class Edge:
    """A data dependency `producer.output[out_slot] -> consumer.input[in_slot]`."""

    producer: int
    consumer: int
    out_slot: int
    in_slot: int

@dataclass
class NodeIR:
    """One ONNX operator, stripped down to what pattern matching needs."""

    id: int
    op_type: str
    domain: str
    attr_key: tuple
    const_input: tuple  # ((slot, dtype_str, shape), ...) for initializer inputs
    output_type: tuple  # ((dtype_str, rank), ...) for the outputs
    n_input: int  # Number of input slots, including the ones fed by initializers
    empty_input_slot: frozenset  # Slots of optional inputs that were left empty
    name: str  # Original ONNX node name, only used for reporting

@dataclass
class GraphIR:
    """Nodes plus adjacency. Immutable as far as the mining stages are concerned."""

    nodes: list[NodeIR]
    edges: list[Edge]
    successor: dict[int, list[Edge]] = field(default_factory=dict)
    predecessor: dict[int, list[Edge]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build the adjacency indices once."""
        self.successor = {n.id: [] for n in self.nodes}
        self.predecessor = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            self.successor[edge.producer].append(edge)
            self.predecessor[edge.consumer].append(edge)

    def label(self, node_id: int, strictness: str) -> tuple:
        """The key two nodes must share to be considered "the same operator"."""
        node = self.nodes[node_id]
        key = (node.op_type, node.domain)
        if strictness in ["L1", "L2", "L3"]:
            key += (node.attr_key, )
        if strictness in ["L2", "L3"]:
            key += (node.const_input, )
        if strictness == "L3":
            key += (node.output_type, )
        return key

MAX_HASHED_ATTRIBUTE_ELEMENT = 4096  # Above this a constant attribute is treated as unique

def _canonical_attribute(value) -> tuple:
    """Turn one graphsurgeon attribute value into something hashable and stable.

    Attribute values are **baked into the function body**, unlike initializer
    inputs which are passed at the call site and may therefore differ between
    instances. So an attribute has to be compared by *value*, not by dtype and
    shape: two `Constant` nodes holding `[2, 16, 64]` and `[8, 16, 16]` have the
    same dtype and shape but are not interchangeable, and merging them silently
    gives every call site the reference instance's constant.

    Very large constants are never merged rather than paying to hash them (and
    to materialise lazily loaded weights).
    """
    if isinstance(value, gs.Graph):
        # A control-flow body. M1 does not mine inside it, but its shape still
        # distinguishes two otherwise identical nodes.
        return ("<graph>", tuple((n.op, n.domain or "") for n in value.nodes))
    if isinstance(value, gs.Constant):
        shape = tuple(value.shape or ())
        n_element = 1
        for dimension in shape:
            n_element *= dimension if isinstance(dimension, int) else MAX_HASHED_ATTRIBUTE_ELEMENT
        if n_element > MAX_HASHED_ATTRIBUTE_ELEMENT:
            return ("<big-constant>", value.name)  # Unique per node, never merges
        return ("<constant>", str(value.dtype), shape, hashlib.md5(np.ascontiguousarray(value.values).tobytes()).hexdigest())
    if isinstance(value, gs.Tensor):
        return ("<tensor>", str(value.dtype), tuple(value.shape or ()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_attribute(v) for v in value)
    if isinstance(value, (bytes, bytearray)):
        return ("<bytes>", bytes(value))
    return value

def _shape_key(tensor) -> tuple:
    """dtype/shape of a tensor without ever materialising its values."""
    shape = tuple(str(d) for d in (tensor.shape or ()))
    return (str(tensor.dtype), shape)

def build_graph_ir(graph: gs.Graph) -> tuple[GraphIR, list[gs.Node]]:
    """Project a graphsurgeon graph into a `GraphIR`.

    Returns the IR and the parallel list of graphsurgeon nodes, so that stage P5
    can map a mined node id back to the object it has to rewrite.
    """
    gs_node_list = list(graph.nodes)
    index_of = {id(node): i for i, node in enumerate(gs_node_list)}

    node_list, edge_list = [], []
    for i, node in enumerate(gs_node_list):
        const_input = []
        for slot, tensor in enumerate(node.inputs):
            if isinstance(tensor, gs.Constant):
                dtype, shape = _shape_key(tensor)
                const_input.append((slot, dtype, shape))
        node_list.append(NodeIR(
            id=i,
            op_type=node.op,
            domain=node.domain or "",
            attr_key=tuple(sorted((k, _canonical_attribute(v)) for k, v in node.attrs.items())),
            const_input=tuple(const_input),
            output_type=tuple((str(t.dtype), len(t.shape) if t.shape is not None else -1) for t in node.outputs),
            n_input=len(node.inputs),
            empty_input_slot=frozenset(s for s, t in enumerate(node.inputs) if t.name == ""),
            name=node.name or f"node_{i}",
        ))

    for i, node in enumerate(gs_node_list):
        for in_slot, tensor in enumerate(node.inputs):
            if tensor.name == "" or isinstance(tensor, gs.Constant):
                continue  # Optional-skipped input, or an initializer (a leaf, not a node)
            for producer in tensor.inputs:  # `Tensor.inputs` are the nodes producing it
                if id(producer) not in index_of:
                    continue
                out_slot = next(k for k, t in enumerate(producer.outputs) if t is tensor)
                edge_list.append(Edge(index_of[id(producer)], i, out_slot, in_slot))

    return GraphIR(node_list, edge_list), gs_node_list
