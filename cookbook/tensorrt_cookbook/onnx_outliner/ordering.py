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
"""The canonical topological order the 1-D reduction stands on.

An *arbitrary* topological order round-robins between parallel regions, which
breaks the "repeated block == repeated substring" equivalence in two ways
(both reproduced in `03-PatternMiningFeasibility/topological_order_experiment.py`):

* a serial chain of blocks running next to an independent branch comes out as
  `ADBDCDADBDCD...` and the pattern is missed entirely;
* two parallel towers come out as `AEBFCGAEBFCG...`, which yields a *false*
  pattern `AEBFCG` -- an interleaving of two unrelated towers.

`recency_topological_sort` emits, among the ready nodes, the one whose latest
predecessor finished most recently, i.e. it finishes the region it is currently
inside before opening a new one.

This is a heuristic, not a guarantee, which is exactly why every candidate is
re-checked in graph space by verify.py.
"""

from __future__ import annotations

from .graph_ir import GraphIR

def depth_first_topological_sort(graph_ir: GraphIR, tie_break: dict[int, tuple] | None = None) -> list[int]:
    """Topological order that finishes the branch it just opened, depth first.

    `recency_topological_sort` below approximates "stay inside the current region"
    by looking at which ready node has the most recently finished predecessor.
    That starves side branches: a node whose only predecessor is the *first* node
    of its block keeps losing both to nodes deeper in the block and to the first
    node of the next block, and ends up at the very end of the order. Measured on
    random planted blocks, that alone scattered a third of the instances::

        B0_0 B0_1 B0_3 B0_4 B1_0 B1_1 B1_3 B1_4 ... Merge B2_2 B1_2 B0_2
                                                            ^^^^^^^^^^^^
                              every block's side branch, pushed past everything

    A plain last-in-first-out ready set does not have that failure: the side
    branch was made ready early, so it is still on the stack and gets popped as
    soon as the deeper path runs out.
    """
    tie_break = tie_break or {}
    in_degree = {node.id: len(graph_ir.predecessor[node.id]) for node in graph_ir.nodes}
    # Sorted descending, because the stack is popped from the end
    ready = sorted([n.id for n in graph_ir.nodes if in_degree[n.id] == 0], key=lambda i: (tie_break.get(i, ()), i), reverse=True)
    order = []
    while ready:
        node_id = ready.pop()
        order.append(node_id)
        newly = []
        for edge in graph_ir.successor[node_id]:
            in_degree[edge.consumer] -= 1
            if in_degree[edge.consumer] == 0:
                newly.append(edge.consumer)
        ready.extend(sorted(newly, key=lambda i: (tie_break.get(i, ()), i), reverse=True))

    if len(order) != len(graph_ir.nodes):
        raise ValueError(f"graph is not a DAG: ordered {len(order)} of {len(graph_ir.nodes)} nodes")
    return order

def recency_topological_sort(graph_ir: GraphIR, tie_break: dict[int, tuple] | None = None) -> list[int]:
    """Topological order that keeps single-entry-single-exit regions contiguous.

    `tie_break` optionally maps a node id to an extra sort key used before the
    node id, e.g. a small-radius structural hash, to make the order depend on
    the graph shape rather than on the order nodes happened to be stored in.
    """
    tie_break = tie_break or {}
    in_degree = {node.id: len(graph_ir.predecessor[node.id]) for node in graph_ir.nodes}
    finish_time = {}

    def sort_key(node_id: int) -> tuple:
        """Latest-finishing predecessor first, then the caller's key, then id."""
        latest = max((finish_time[e.producer] for e in graph_ir.predecessor[node_id]), default=-1)
        return (-latest, tie_break.get(node_id, ()), node_id)

    # A plain re-sorted list rather than a heap: the key of every waiting node
    # changes whenever a node finishes, so a heap would only hold stale keys.
    # The ready set stays tiny on real graphs, so this is cheap in practice.
    ready = [n.id for n in graph_ir.nodes if in_degree[n.id] == 0]
    order = []
    while ready:
        ready.sort(key=sort_key)
        node_id = ready.pop(0)
        order.append(node_id)
        finish_time[node_id] = len(order)
        for edge in graph_ir.successor[node_id]:
            in_degree[edge.consumer] -= 1
            if in_degree[edge.consumer] == 0:
                ready.append(edge.consumer)

    if len(order) != len(graph_ir.nodes):
        raise ValueError(f"graph is not a DAG: ordered {len(order)} of {len(graph_ir.nodes)} nodes")
    return order
