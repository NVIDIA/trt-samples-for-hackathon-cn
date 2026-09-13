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
"""Does a parallel branch break the "repeated substring of the topological order" trick?

`graph_statistics.py` showed that a 6-layer TransformerEncoder flattens into a
perfectly periodic op-type string. That model is almost a chain, so it is the
easy case. This script builds the three structures that are *not* the easy case
and checks two orderings:

* `nx.topological_sort`   - an arbitrary valid topological order
* `recency_topological_sort` - greedy "stay on the branch you are already on",
  i.e. among the ready nodes always emit the one whose predecessor finished most
  recently. This keeps a single-entry-single-exit region contiguous instead of
  round-robining between parallel regions.

The conclusion drives the design in DESIGN.md: the 1-D reduction is only sound
on top of a *canonical* topological order, and even then every candidate must be
verified in graph space.
"""

from collections import defaultdict

import networkx as nx

def recency_topological_sort(graph: nx.DiGraph) -> list:
    """Topological order that prefers to finish the region it is currently inside."""
    index_of = {n: i for i, n in enumerate(graph.nodes)}
    in_degree = {n: graph.in_degree(n) for n in graph}
    ready = [n for n in graph if in_degree[n] == 0]
    finish_time, order, clock = {}, [], 0
    while ready:
        # Latest-finishing predecessor first, then original node order for determinism
        ready.sort(key=lambda n: (-max((finish_time[p] for p in graph.predecessors(n)), default=-1), index_of[n]))
        node = ready.pop(0)
        order.append(node)
        finish_time[node] = clock
        clock += 1
        for successor in graph.successors(node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                ready.append(successor)
    return order

def longest_repeat(label_list: list, n_repeat_min: int) -> tuple:
    """Longest contiguous sub-sequence repeating >= n_repeat_min times without overlap."""
    best = None
    for length in range(2, max(3, len(label_list) // n_repeat_min + 1)):
        position_dict = defaultdict(list)
        for i in range(len(label_list) - length + 1):
            position_dict[tuple(label_list[i:i + length])].append(i)
        found = False
        for key, position_list in position_dict.items():
            kept, last = [], -len(label_list)
            for p in position_list:
                if p - last >= length:
                    kept.append(p)
                    last = p
            if len(kept) >= n_repeat_min:
                best, found = (length, key, kept), True
                break
        if not found:
            break
    return best

# ================================================================ Test structures

def build_serial_plus_parallel_chain() -> nx.DiGraph:
    """4 copies of A->B->C in series, plus a completely independent chain of D."""
    graph = nx.DiGraph()
    previous = None
    for i in range(4):
        for op in "ABC":
            name = f"{op}{i}"
            graph.add_node(name, op_type=op)
            if previous is not None:
                graph.add_edge(previous, name)
            previous = name
    previous = None
    for j in range(6):
        name = f"D{j}"
        graph.add_node(name, op_type="D")
        if previous is not None:
            graph.add_edge(previous, name)
        previous = name
    return graph

def build_block_with_internal_branch() -> nx.DiGraph:
    """4 copies of a block that itself forks into 3 parallel branches (Q/K/V style)."""
    graph = nx.DiGraph()
    previous = None
    for i in range(4):
        source = f"S{i}"
        graph.add_node(source, op_type="S")
        if previous is not None:
            graph.add_edge(previous, source)
        for branch in "QKV":
            name = f"{branch}{i}"
            graph.add_node(name, op_type=branch)
            graph.add_edge(source, name)
        merge = f"M{i}"
        graph.add_node(merge, op_type="M")
        for branch in "QKV":
            graph.add_edge(f"{branch}{i}", merge)
        previous = merge
    return graph

def build_two_tower() -> nx.DiGraph:
    """Two independent towers, each with its own 4 repeated blocks."""
    graph = nx.DiGraph()
    for op_set in ["ABC", "EFG"]:
        previous = None
        for i in range(4):
            for op in op_set:
                name = f"{op}{i}"
                graph.add_node(name, op_type=op)
                if previous is not None:
                    graph.add_edge(previous, name)
                previous = name
    return graph

# ================================================================ Entrance

def report(title: str, graph: nx.DiGraph, n_repeat_min: int, expected: str) -> None:
    """Run both orderings on one structure and print what the 1-D search finds."""
    print(f"=== {title}")
    print(f"    expected pattern: {expected}")
    for name, order in [
        ("arbitrary", list(nx.topological_sort(graph))),
        ("recency  ", recency_topological_sort(graph)),
    ]:
        label_list = [graph.nodes[n]["op_type"] for n in order]
        best = longest_repeat(label_list, n_repeat_min)
        text = f"length={best[0]}, pattern={''.join(best[1])}, at {best[2]}" if best else "NOTHING FOUND"
        print(f"    [{name}] sequence = {''.join(label_list)}")
        print(f"    {' ' * (len(name) + 2)} found    = {text}")
    return

if __name__ == "__main__":
    report("1. serial blocks + an independent parallel chain", build_serial_plus_parallel_chain(), 4, "ABC x4")
    print()
    report("2. blocks with internal parallel Q/K/V branches", build_block_with_internal_branch(), 4, "SQKVM x4")
    print()
    report("3. two parallel towers, each with its own repeated blocks", build_two_tower(), 4, "ABC x4 and EFG x4")

    print("\n" + "=" * 30 + " Conclusion")
    print("1. An arbitrary topological order interleaves parallel regions and the pattern is MISSED.")
    print("2. Parallelism *inside* a block is harmless: the block stays contiguous either way.")
    print("3. Worse than missing: an arbitrary order can report a FALSE pattern (`AEBFCG` in case 3),")
    print("   an interleaving of two unrelated towers that is not a meaningful module at all.")
    print("-> The 1-D reduction needs a canonical, region-preserving topological order,")
    print("   and every candidate must still be verified by real sub-graph isomorphism.")
    print("\nFinish")
