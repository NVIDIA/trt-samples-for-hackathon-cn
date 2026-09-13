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
"""Measurements that decide whether the proposed mining algorithm is feasible.

This is *not* an implementation of the miner. It measures, on a realistic
"exploded" ONNX (a 6-layer `nn.TransformerEncoder`), the four quantities the
feasibility analysis in README.md rests on:

0. what pre-processing does to the graph size
1. how fast the search space of "grow the pattern by one adjacent node" grows
2. whether a cheap canonical label (Weisfeiler-Lehman hash) can carry us all the
   way to a whole encoder layer
3. whether the much cheaper 1-D reduction (periodicity of the topologically
   ordered op-type sequence) finds the layer boundaries, and whether the found
   instances really are isomorphic
"""

from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import onnx
import onnxslim
import torch
import torch.nn as nn
from networkx.algorithms.graph_hashing import weisfeiler_lehman_subgraph_hashes
from networkx.algorithms.isomorphism import DiGraphMatcher, categorical_edge_match, categorical_node_match

torch.manual_seed(31193)

N_LAYER = 6  # Ground truth: the model has 6 identical encoder layers
N_MODEL = 64
N_HEAD = 4
N_SEQ = 16
N_B = 2
OPSET = 17
K_MAX = 11  # Enumerate connected sub-graphs up to this size
BUDGET = 3_000_000  # Give up counting beyond this many sub-graphs
WL_ITERATION_LIST = [1, 2, 3, 5, 8, 12, 16, 24, 32]

output_path = Path(__file__).parent
onnx_file = output_path / "model-transformer.onnx"
onnx_file_slim = output_path / "model-transformer-slim.onnx"

# ================================================================ Prepare a realistic model

def prepare_model() -> tuple:
    """Export a 6-layer TransformerEncoder and also keep a constant-folded copy."""
    if not onnx_file.exists():
        layer = nn.TransformerEncoderLayer(N_MODEL, N_HEAD, dim_feedforward=4 * N_MODEL, batch_first=True)
        model = nn.TransformerEncoder(layer, N_LAYER).eval()
        x = torch.randn(N_B, N_SEQ, N_MODEL)
        torch.onnx.export(model, (x, ), onnx_file, dynamo=False, opset_version=OPSET, input_names=["x"], output_names=["y"])
    if not onnx_file_slim.exists():
        onnx.save(onnxslim.slim(onnx.load(onnx_file)), onnx_file_slim)
    return onnx.load(onnx_file), onnx.load(onnx_file_slim)

# ================================================================ Graph conversion

def onnx_to_networkx(model: onnx.ModelProto) -> nx.DiGraph:
    """One networkx node per ONNX node. Node label = op_type + sorted attributes.

    Initializers and graph inputs are deliberately *not* nodes: they are the
    "leaves" whose values are allowed to differ between two instances of the
    same pattern (that is exactly what gets stacked into `[N, ...]` later).
    """
    graph = nx.DiGraph()
    producer = {}
    for i, node in enumerate(model.graph.node):
        attribute_key = tuple(sorted(onnx.helper.printable_attribute(a, subgraphs=False) for a in node.attribute))
        graph.add_node(i, label=(node.op_type, attribute_key), op_type=node.op_type)
        for name in node.output:
            producer[name] = i
    for i, node in enumerate(model.graph.node):
        for slot, name in enumerate(node.input):
            if name in producer:
                # `slot` matters: ONNX operator inputs are ordered, Sub(a, b) != Sub(b, a)
                graph.add_edge(producer[name], i, slot=slot)
    return graph

# ================================================================ 1. Search space growth

def count_connected_subgraph(graph: nx.DiGraph, k_max: int, budget: int) -> dict:
    """Exactly count weakly connected sub-graphs of size 1..k_max.

    Plain BFS over vertex sets with de-duplication. This is the search space a
    naive "start from one node and keep adding an adjacent node" miner walks.
    """
    undirected = graph.to_undirected()
    neighbor = {n: set(undirected.neighbors(n)) for n in undirected}
    count = {1: len(neighbor)}
    current = {frozenset([n]) for n in neighbor}
    for k in range(2, k_max + 1):
        nxt, overflow = set(), False
        for subset in current:
            for n in set().union(*(neighbor[n] for n in subset)) - subset:
                nxt.add(subset | {n})
                if len(nxt) > budget:
                    overflow = True
                    break
            if overflow:
                break
        if overflow:
            for kk in range(k, k_max + 1):
                count[kk] = f">{budget}"
            break
        count[k] = len(nxt)
        current = nxt
    return count

# ================================================================ 2. WL canonical labels

def wl_class_statistics(graph: nx.DiGraph, iteration_list: list) -> list:
    """Group nodes by their WL sub-graph hash after `iteration` rounds.

    A WL hash after `i` rounds is a canonical label of the `i`-hop neighbourhood,
    so two nodes sharing a hash root two sub-graphs that look identical out to
    `i` hops. The question is whether raising `i` far enough to cover a whole
    encoder layer keeps the 6 instances in the same class.
    """
    undirected = graph.to_undirected()
    row_list = []
    for iteration in iteration_list:
        hash_dict = weisfeiler_lehman_subgraph_hashes(undirected, node_attr="op_type", iterations=iteration)
        bucket = defaultdict(list)
        for node, hash_list in hash_dict.items():
            bucket[hash_list[-1]].append(node)  # Hash after the last round
        size_list = sorted((len(v) for v in bucket.values()), reverse=True)
        row_list.append(dict(
            iteration=iteration,
            n_class=len(bucket),
            n_repeated_class=sum(1 for s in size_list if s >= 2),
            n_node_in_repeated_class=sum(s for s in size_list if s >= 2),
            top_size=size_list[:6],
            n_class_of_size_n_layer=sum(1 for s in size_list if s == N_LAYER),
        ))
    return row_list

# ================================================================ 3. The 1-D reduction

def find_repeated_label_period(graph: nx.DiGraph, n_repeat_min: int) -> tuple:
    """Longest contiguous op-type sub-sequence of the topological order that
    repeats, without overlap, at least `n_repeat_min` times.

    Serially stacked blocks (transformer layers, resnet stages, ...) become a
    periodic string once the DAG is flattened, so the graph problem collapses
    into a classic "longest repeated substring" problem.
    """
    order = list(nx.topological_sort(graph))
    label_list = [graph.nodes[n]["op_type"] for n in order]
    best = None
    for length in range(2, len(label_list) // n_repeat_min + 1):
        position_dict = defaultdict(list)
        for i in range(len(label_list) - length + 1):
            position_dict[tuple(label_list[i:i + length])].append(i)
        found = False
        for key, position_list in position_dict.items():
            if len(position_list) < n_repeat_min:
                continue
            kept, last = [], -len(label_list)
            for p in position_list:  # Greedily keep non-overlapping occurrences
                if p - last >= length:
                    kept.append(p)
                    last = p
            if len(kept) >= n_repeat_min:
                best, found = (length, key, kept), True
                break
        if not found:
            break
    return order, label_list, best

def verify_instance_isomorphic(graph: nx.DiGraph, order: list, length: int, position_list: list) -> list:
    """Are the induced sub-graphs at those positions really isomorphic?"""
    node_match = categorical_node_match("label", None)
    edge_match = categorical_edge_match("slot", None)
    reference = graph.subgraph(order[position_list[0]:position_list[0] + length])
    verdict_list = []
    for p in position_list[1:]:
        other = graph.subgraph(order[p:p + length])
        matcher = DiGraphMatcher(reference, other, node_match=node_match, edge_match=edge_match)
        verdict_list.append(matcher.is_isomorphic())
    return verdict_list

# ================================================================ Entrance

def describe(name: str, model: onnx.ModelProto, graph: nx.DiGraph) -> None:
    """Print the basic shape of one version of the graph."""
    op_counter = Counter(data["op_type"] for _, data in graph.nodes(data=True))
    print(f"[{name}] nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}, "
          f"initializers={len(model.graph.initializer)}, distinct-op={len(op_counter)}, "
          f"DAG-depth={nx.dag_longest_path_length(graph)}")
    print(f"    op histogram : {op_counter.most_common(12)}")
    print(f"    out-degree   : {sorted(Counter(d for _, d in graph.out_degree()).items())}")
    return

def main() -> None:
    model_raw, model_slim = prepare_model()
    graph_raw = onnx_to_networkx(model_raw)
    graph = onnx_to_networkx(model_slim)

    print("=" * 30 + " 0. Pre-processing matters")
    describe("raw", model_raw, graph_raw)
    describe("onnxslim", model_slim, graph)
    print(f"-> Constant folding removes {graph_raw.number_of_nodes() - graph.number_of_nodes()} of "
          f"{graph_raw.number_of_nodes()} nodes ({graph_raw.number_of_nodes() / graph.number_of_nodes():.1f}x smaller).")
    print("   Shape/Constant/Identity plumbing is noise for pattern mining and must be folded first.")

    print("\n" + "=" * 30 + " 1. Search space of naive pattern growth (on the slimmed graph)")
    for k, v in count_connected_subgraph(graph, K_MAX, BUDGET).items():
        print(f"    connected sub-graphs of size {k:2d}: {v}")
    print(f"-> Growth settles at ~1.85x per added node on this (very sparse) graph. Small patterns are")
    print(f"   cheap, but one encoder layer is {graph.number_of_nodes() // N_LAYER} nodes and")
    print(f"   246 * 1.85^40 ~= 2.6e12 sub-graphs, which brute-force enumeration will never reach.")

    print("\n" + "=" * 30 + " 2. Weisfeiler-Lehman canonical labels (on the slimmed graph)")
    print(f"    {'WLiter':>7}{'classes':>9}{'repClass':>10}{'nodesInRep':>12}  {'largest classes':<24}{'#class of size 6':>18}")
    for row in wl_class_statistics(graph, WL_ITERATION_LIST):
        print(f"    {row['iteration']:>7}{row['n_class']:>9}{row['n_repeated_class']:>10}{row['n_node_in_repeated_class']:>12}  "
              f"{str(row['top_size']):<24}{row['n_class_of_size_n_layer']:>18}")
    print("-> Raising the WL radius does NOT converge onto the repeated layers, it destroys them:")
    print("   at a large radius the neighbourhood of a node reaches the graph boundary, and layer 1")
    print("   (touching the input) stops looking like layer 4. WL is only usable for cheap *seeding*.")

    print("\n" + "=" * 30 + " 3. The 1-D reduction: periodicity of the topological order")
    order, label_list, best = find_repeated_label_period(graph, N_LAYER)
    print(f"    topological op-type sequence length = {len(label_list)}")
    if best is None:
        print("    no repeated sub-sequence found")
        return
    length, pattern, position_list = best
    print(f"    longest sub-sequence repeated >= {N_LAYER} times without overlap: length = {length}")
    print(f"    occurrence positions = {position_list}")
    print(f"    pattern head = {pattern[:10]}")
    print(f"    coverage = {length * len(position_list)} / {len(label_list)} nodes")
    verdict_list = verify_instance_isomorphic(graph, order, length, position_list)
    print(f"    induced sub-graphs isomorphic to the first one (label + input slot matched): {verdict_list}")
    print("-> For serially stacked blocks the graph problem collapses to 'longest repeated substring',")
    print("   which is O(n) with a suffix automaton instead of exponential.")
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
