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
"""P2 path B -- graph-space search, for blocks the 1-D path cannot see whole.

`stress_test.py` in `07-ParallelRepeat` measures what path A actually misses on
random planted blocks. It is not "parallel repetition is invisible", which is
what DESIGN.md assumed: it is that a filler node lands in the middle of a block,
the block stops being a contiguous run of the topological order, and it comes
back **split into two patterns** or short a boundary node.

Path B does not depend on contiguity at all, but it must not fall into the trap
the original proposal did:

* enumerating candidate patterns is hopeless -- the search space grows ~1.85x
  per node and one encoder layer is 41 nodes;
* so nothing is enumerated. Cheap small-radius Weisfeiler-Lehman hashes give
  *anchor groups* (nodes that look alike a few hops out), and each group is then
  grown **in lockstep**: at every step all instances extend along the same edge
  or nobody does. Growth is driven by agreement, not by branching, so there is
  exactly one successor state per step instead of many.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

from .graph_ir import GraphIR

def weisfeiler_lehman(graph_ir: GraphIR, strictness: str, radius: int) -> dict:
    """Colour every node by the shape of its `radius`-hop neighbourhood.

    Slots are part of the message, so `Sub(a, b)` and `Sub(b, a)` never get the
    same colour. The radius stays small on purpose: a large one reaches the
    graph boundary and starts telling apart blocks that *are* interchangeable
    (measured in `03-PatternMiningFeasibility/graph_statistics.py`: past 12
    rounds not a single class of size 6 survives on a 6-layer encoder).
    """
    colour = {node.id: repr(graph_ir.label(node.id, strictness)) for node in graph_ir.nodes}
    for _ in range(radius):
        nxt = {}
        for node in graph_ir.nodes:
            outgoing = sorted((e.out_slot, e.in_slot, colour[e.consumer]) for e in graph_ir.successor[node.id])
            incoming = sorted((e.in_slot, e.out_slot, colour[e.producer]) for e in graph_ir.predecessor[node.id])
            nxt[node.id] = hashlib.md5(repr((colour[node.id], outgoing, incoming)).encode()).hexdigest()[:16]
        colour = nxt
    return colour

def _edge_list(graph_ir: GraphIR, node_id: int, colour: dict) -> list:
    """Every edge of one node, in a canonical order, tagged with its key.

    The key is deliberately built from the **plain node label**, never from a WL
    colour: a WL colour folds in the neighbourhood *outside* the block, and two
    instances of the same block normally hang off different tensors, so their
    colours differ and no two instances would ever agree on an edge. Growth has
    to treat everything beyond the current region as a wildcard.
    """
    result = []
    for edge in graph_ir.successor[node_id]:
        result.append(((True, edge.out_slot, edge.in_slot, colour[edge.consumer]), True, edge))
    for edge in graph_ir.predecessor[node_id]:
        result.append(((False, edge.out_slot, edge.in_slot, colour[edge.producer]), False, edge))
    result.sort(key=lambda item: item[0])
    return result

def _other_end(edge, b_outgoing: bool) -> int:
    """The node at the far side of an edge."""
    return edge.consumer if b_outgoing else edge.producer

def _local_signature(graph_ir: GraphIR, node_id: int, colour: dict) -> tuple:
    """Enough of a node's own shape to tell same-labelled siblings apart.

    A block often hands the same value to two nodes of the same operator, e.g.
    `B_0 -> B_1` and `B_0 -> B_2` where both are `Relu`. `(direction, slot, slot,
    label)` cannot separate those two, and refusing to grow through an ambiguous
    edge is what left the side branch of a third of the random blocks behind.
    Their own edge sets do separate them.
    """
    return (colour[node_id], tuple(key for key, _, _ in _edge_list(graph_ir, node_id, colour)))

def _paired_edges(graph_ir: GraphIR, node_list: list, colour: dict, reference_offset_node: int) -> list | None:
    """Line up the edges of every instance's node with the reference's, key by key.

    Returns a list of `(b_outgoing, [neighbour per instance])`, or None when some
    instance does not present the same edges. Ties inside one key group are broken
    by `_local_signature`; if that still ties, the pairing is arbitrary and a wrong
    guess simply fails `verify` later, so it costs a candidate, never correctness.
    """
    group = {}
    for key, b_outgoing, edge in _edge_list(graph_ir, reference_offset_node, colour):
        group.setdefault((key, b_outgoing), []).append(edge)

    result = []
    for (key, b_outgoing), reference_edge in group.items():
        per_instance = []
        for node_id in node_list:
            match = [e for k, o, e in _edge_list(graph_ir, node_id, colour) if k == key and o == b_outgoing]
            if len(match) != len(reference_edge):
                per_instance = None
                break
            per_instance.append(sorted(match, key=lambda e: _local_signature(graph_ir, _other_end(e, b_outgoing), colour)))
        if per_instance is None:
            continue  # This edge is not present in the same multiplicity everywhere
        for slot in range(len(reference_edge)):
            result.append((b_outgoing, [_other_end(per_instance[i][slot], b_outgoing) for i in range(len(node_list))]))
    return result

def grow_in_lockstep(graph_ir: GraphIR, seed: list, colour: dict, max_size: int) -> list | None:
    """Extend an aligned group of regions, all along the same edge or none at all.

    `seed` is a list of aligned node-id lists (one per instance); a list of
    single-element lists starts from bare anchors. Returns the grown group, or
    None if the group fell apart.
    """
    member = [list(node_list) for node_list in seed]
    member_set = [set(node_list) for node_list in member]
    taken = {node_id for node_list in member for node_id in node_list}
    if len(taken) != sum(len(node_list) for node_list in member):
        return None  # The seed instances already overlap

    offset = 0
    while offset < len(member[0]) and len(member[0]) < max_size:
        for _, candidate in _paired_edges(graph_ir, [node_list[offset] for node_list in member], colour, member[0][offset]):
            already = [node_id in member_set[i] for i, node_id in enumerate(candidate)]
            if all(already):
                # Must already sit at the same offset everywhere, otherwise the
                # instances are not aligned and the group is unusable.
                position = {member[i].index(node_id) for i, node_id in enumerate(candidate)}
                if len(position) != 1:
                    return None
                continue
            if any(already):
                return None  # Inside for some instances, outside for others
            if len(set(candidate)) != len(candidate) or any(node_id in taken for node_id in candidate):
                continue  # Would make two instances overlap

            for i, node_id in enumerate(candidate):
                member[i].append(node_id)
                member_set[i].add(node_id)
                taken.add(node_id)
        offset += 1

    return member

def _label_colour(graph_ir: GraphIR, strictness: str) -> dict:
    """The plain per-node label, which is what growth compares neighbours by."""
    return {node.id: repr(graph_ir.label(node.id, strictness)) for node in graph_ir.nodes}

def _canonical_order(member: list, position: dict) -> list:
    """Sort the reference instance topologically, permuting the others the same way.

    The function body is built from the reference instance, so that one has to
    come out topologically sorted; applying the identical permutation everywhere
    keeps the instances aligned.
    """
    permutation = sorted(range(len(member[0])), key=lambda k: position[member[0][k]])
    return [[node_list[k] for k in permutation] for node_list in member]

def extend_group(graph_ir: GraphIR, order: list, strictness: str, instance_node_list: list, max_size: int, skip: set) -> list | None:
    """Grow an already aligned group of instances, e.g. one the 1-D search found.

    This is the workhorse. `stress_test.py` shows the 1-D search rarely misses a
    block outright; what it does is report it **truncated or split in two**,
    because a foreign node landed in the middle of the run. Seeding growth with
    that partial answer recovers the rest, and needs no anchor hashing at all.
    """
    colour = _label_colour(graph_ir, strictness)
    position = {node_id: i for i, node_id in enumerate(order)}
    member = grow_in_lockstep(graph_ir, instance_node_list, colour, max_size)
    if member is None or len(member[0]) <= len(instance_node_list[0]):
        return None  # Fell apart, or nothing was added
    if any(node_id in skip for node_list in member for node_id in node_list):
        return None
    return _canonical_order(member, position)

def _match_instance(graph_ir: GraphIR, reference: list, anchor: int, colour: dict, taken: set) -> list | None:
    """Try to lay the reference instance over the graph starting at `anchor`.

    Unlike growth, the shape is already known, so this just walks the reference's
    **internal** edges and follows the same edge from the candidate. Edges that
    leave the reference are wildcards: two instances of one block normally hang
    off completely different tensors.
    """
    offset_of = {node_id: k for k, node_id in enumerate(reference)}
    mapped = {0: anchor}
    used = {anchor}
    if anchor in taken or colour[anchor] != colour[reference[0]]:
        return None

    for offset in range(len(reference)):
        if offset not in mapped:
            return None  # The reference is not connected from node 0 in this direction
        # Pairing the reference against itself and the candidate at the same time
        # lines the two edge sets up, disambiguating same-labelled siblings by
        # their own edge sets.
        for _, (other, candidate) in _paired_edges(graph_ir, [reference[offset], mapped[offset]], colour, reference[offset]):
            if other not in offset_of:
                continue  # Leaves the instance, wildcard
            target_offset = offset_of[other]
            if target_offset in mapped:
                if mapped[target_offset] != candidate:
                    return None  # Inconsistent with what we already mapped
            elif candidate in taken or candidate in used:
                return None
            else:
                mapped[target_offset] = candidate
                used.add(candidate)

    if len(mapped) != len(reference):
        return None
    return [mapped[k] for k in range(len(reference))]

def find_more_instances(graph_ir: GraphIR, order: list, strictness: str, instance_node_list: list, skip: set) -> list:
    """Occurrences of an already accepted pattern that the search did not report.

    The dominant failure of the 1-D reduction is not a wrong block shape, it is
    **one instance short**: an occurrence whose nodes are not a contiguous run of
    the topological order never becomes a candidate window at all. Once the shape
    is known, finding the rest is a cheap structural match.
    """
    colour = _label_colour(graph_ir, strictness)
    reference = instance_node_list[0]
    taken = set(skip) | {node_id for node_list in instance_node_list for node_id in node_list}
    position = {node_id: i for i, node_id in enumerate(order)}

    extra = []
    for node in graph_ir.nodes:
        if node.id in taken:
            continue
        matched = _match_instance(graph_ir, reference, node.id, colour, taken)
        if matched is None:
            continue
        extra.append(matched)
        taken.update(matched)
    return sorted(extra, key=lambda node_list: position[node_list[0]])

def find_parallel_candidates(graph_ir: GraphIR, order: list, strictness: str, radius: int, min_repeat: int, min_size: int, max_size: int, skip: set) -> list:
    """Anchor groups grown from scratch, sorted by decreasing gain.

    Used when the 1-D search has nothing to offer. Anchors are bucketed by a
    small-radius Weisfeiler-Lehman hash, which *does* fold in outside context and
    therefore only groups instances that also sit in similar surroundings. That
    is a real limitation, and the reason `extend_group` above carries most of the
    weight; this path is the fallback for blocks the 1-D search never saw.
    """
    wl_colour = weisfeiler_lehman(graph_ir, strictness, radius)
    bucket = defaultdict(list)
    for node in graph_ir.nodes:
        if node.id not in skip:
            bucket[wl_colour[node.id]].append(node.id)

    colour = _label_colour(graph_ir, strictness)
    position = {node_id: i for i, node_id in enumerate(order)}
    candidate_list = []
    for anchor_list in bucket.values():
        if len(anchor_list) < min_repeat:
            continue
        seed = [[anchor] for anchor in sorted(anchor_list, key=lambda n: position[n])]
        member = grow_in_lockstep(graph_ir, seed, colour, max_size)
        if member is None or len(member[0]) < min_size:
            continue
        if any(node_id in skip for node_list in member for node_id in node_list):
            continue
        candidate_list.append(_canonical_order(member, position))

    return sorted(candidate_list, key=lambda m: -((len(m[0]) - 1) * (len(m) - 1)))
