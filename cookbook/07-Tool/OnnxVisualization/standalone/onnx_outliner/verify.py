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
"""P3 -- everything a candidate has to survive before it may be outlined.

The 1-D search only compares op labels, so it can hand us garbage. Three checks
run in graph space, in increasing cost order:

1. **convexity** -- contracting an instance must not create a cycle, otherwise
   "compute part of the block, leave, come back" cannot be one function call.
2. **alignment** -- instance `j`'s node at offset `t` must play exactly the role
   of instance `0`'s node at offset `t`, edges and slots included. We do not
   *search* for a mapping (VF2 style) because the mapping we will actually build
   the function body with is the positional one; if that one fails the candidate
   is useless even if some other mapping exists.
3. **interface** -- all instances must expose the same external inputs and
   outputs, at the same offsets and slots.
"""

from __future__ import annotations

from dataclasses import dataclass

from .graph_ir import GraphIR

@dataclass
class Instance:
    """One occurrence of a pattern: node ids in canonical (aligned) order."""

    node_id_list: list[int]
    external_input: list[tuple]  # (offset, in_slot) of each external input, first-use order
    external_output: list[tuple]  # (offset, out_slot) of each tensor consumed outside

@dataclass
class VerifiedPattern:
    """A candidate that survived P3."""

    length: int
    instance_list: list[Instance]
    interface_signature: tuple

    @property
    def gain(self) -> int:
        """Nodes removed from the main graph."""
        return (self.length - 1) * (len(self.instance_list) - 1)

def is_convex(graph_ir: GraphIR, node_id_list: list[int]) -> bool:
    """Can this node set be contracted into a single node without making a cycle?

    Equivalent formulation, cheaper than contracting: there must be no path that
    leaves the set and comes back. Walk forward from every successor that is
    outside the set and see whether it can re-enter.
    """
    inside = set(node_id_list)
    stack = [e.consumer for n in node_id_list for e in graph_ir.successor[n] if e.consumer not in inside]
    seen = set(stack)
    while stack:
        node_id = stack.pop()
        for edge in graph_ir.successor[node_id]:
            if edge.consumer in inside:
                return False  # Left the set and came back
            if edge.consumer not in seen:
                seen.add(edge.consumer)
                stack.append(edge.consumer)
    return True

def _internal_edge_set(graph_ir: GraphIR, node_id_list: list[int]) -> set:
    """Edges with both ends inside the instance, expressed in offsets."""
    offset_of = {node_id: i for i, node_id in enumerate(node_id_list)}
    return {(offset_of[e.producer], offset_of[e.consumer], e.out_slot, e.in_slot) for n in node_id_list for e in graph_ir.successor[n] if e.consumer in offset_of}

def _interface(graph_ir: GraphIR, node_id_list: list[int], graph_output_node: set) -> Instance:
    """External inputs and outputs of one instance, expressed as (offset, slot).

    An input slot is external when nothing inside the instance produces it, no
    matter whether it is fed by an initializer, a graph input or an outside node.
    Slots are not de-duplicated: if the same tensor feeds two slots it simply
    becomes two function inputs, which stays correct because the call site
    passes the real tensors.

    An output is external when something outside consumes it, or when it is a
    graph output. Those become extra function outputs rather than a reason to
    reject the candidate.
    """
    inside = set(node_id_list)

    external_input = []
    for offset, node_id in enumerate(node_id_list):
        node = graph_ir.nodes[node_id]
        incoming = {e.in_slot: e for e in graph_ir.predecessor[node_id]}
        for in_slot in range(node.n_input):
            if in_slot in node.empty_input_slot:
                continue  # Optional input left empty, stays empty inside the function
            edge = incoming.get(in_slot)
            if edge is not None and edge.producer in inside:
                continue  # Produced inside the instance
            external_input.append((offset, in_slot))

    external_output = []
    for offset, node_id in enumerate(node_id_list):
        for out_slot in range(len(graph_ir.nodes[node_id].output_type)):
            leaves = any(e.out_slot == out_slot and e.consumer not in inside for e in graph_ir.successor[node_id])
            if leaves or (node_id, out_slot) in graph_output_node:
                external_output.append((offset, out_slot))

    return Instance(list(node_id_list), external_input, external_output)

def verify(
    graph_ir: GraphIR,
    instance_node_list: list[list[int]],
    graph_output_node: set,
    reject_counter: dict,
) -> VerifiedPattern | None:
    """Run the three checks. Return None (and count the reason) on failure."""
    if len(instance_node_list) < 2:
        return None

    for node_id_list in instance_node_list:
        if not is_convex(graph_ir, node_id_list):
            reject_counter["non_convex"] = reject_counter.get("non_convex", 0) + 1
            return None

    reference_edge = _internal_edge_set(graph_ir, instance_node_list[0])
    for node_id_list in instance_node_list[1:]:
        if _internal_edge_set(graph_ir, node_id_list) != reference_edge:
            reject_counter["alignment_mismatch"] = reject_counter.get("alignment_mismatch", 0) + 1
            return None

    instance_list = [_interface(graph_ir, node_id_list, graph_output_node) for node_id_list in instance_node_list]
    signature_set = {(tuple(i.external_input), tuple(i.external_output)) for i in instance_list}
    if len(signature_set) != 1:
        reject_counter["interface_mismatch"] = reject_counter.get("interface_mismatch", 0) + 1
        return None

    return VerifiedPattern(len(instance_node_list[0]), instance_list, signature_set.pop())
