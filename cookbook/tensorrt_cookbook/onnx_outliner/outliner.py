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
"""The driver: P0 -> P1 -> P2 -> P3 -> P4 -> P5 -> P6."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from .config import OutlineConfig
from .discover import find_candidates
from .discover_parallel import extend_group, find_more_instances, find_parallel_candidates
from .graph_ir import build_graph_ir
from .ordering import depth_first_topological_sort, recency_topological_sort
from .rewrite import outline_pattern
from .rewrite_loop import analyse as analyse_loop
from .rewrite_loop import outline_pattern_as_loop
from .verify import verify

def sort_nodes_in_place(graph: gs.Graph) -> None:
    """Topologically order `graph.nodes`, sub-graph dependencies included.

    Not `gs.Graph.toposort()`, which recurses forever as soon as a graph has both
    local functions and a `Loop`/`If` body (onnx_graphsurgeon 0.6.1, the inner
    `get_used_funcs` iterates `self.subgraphs()` instead of the node list it was
    handed). Outlining inside a sub-graph produces exactly that combination.

    A node that owns a sub-graph also depends on whatever the body reads from the
    enclosing scope, which is not in the node's own input list; missing those
    would emit the node before its producer.
    """
    produced = {id(tensor): node for node in graph.nodes for tensor in node.outputs}
    dependency = {}
    for node in graph.nodes:
        needed = list(node.inputs)
        for subgraph in node.subgraphs(recursive=True):
            needed.extend(subgraph._foreign_tensors().values())
        dependency[id(node)] = {id(produced[id(t)]) for t in needed if id(t) in produced and produced[id(t)] is not node}

    remaining = {id(node): node for node in graph.nodes}
    done, order = set(), []
    while remaining:
        ready = [key for key, node in remaining.items() if dependency[key] <= done]
        if not ready:
            raise ValueError("graph is not a DAG")
        for key in ready:
            order.append(remaining.pop(key))
        done.update(ready)
    graph.nodes.clear()
    graph.nodes.extend(order)
    return

def _graph_output_node(graph: gs.Graph, gs_node_list: list) -> set:
    """`(node_id, out_slot)` pairs that are also graph outputs."""
    output_tensor = {id(t) for t in graph.outputs}
    result = set()
    for i, node in enumerate(gs_node_list):
        for out_slot, tensor in enumerate(node.outputs):
            if id(tensor) in output_tensor:
                result.add((i, out_slot))
    return result

def _is_homogeneous_run(gs_node_list: list, node_id_list: list) -> bool:
    """Is this instance just N copies of the same operator in a row?

    Grouping a uniform run adds no information. Six identical encoder layers
    have no natural "two groups of three", and reporting one implies a structure
    the model does not have, so a *nested* pattern of this shape is refused.
    At level 0 it is fine: a run of six `Softplus` really is a repeated block.
    """
    if len(node_id_list) < 2:
        return False
    key = {(gs_node_list[i].op, gs_node_list[i].domain) for i in node_id_list}
    return len(key) == 1

def _find_patterns(graph: gs.Graph, config: OutlineConfig, reject_counter: dict, b_nested: bool = False) -> tuple:
    """P1..P4: everything that happens before the graph is touched.

    Runs "find the best candidate, verify it, commit it, mask it out, repeat"
    so that a model with several different repeated blocks (a two-tower model,
    a network with several resnet stages) yields several patterns.
    """
    graph_ir, gs_node_list = build_graph_ir(graph)
    graph_output_node = _graph_output_node(graph, gs_node_list)
    order = (depth_first_topological_sort if config.ordering == "dfs" else recency_topological_sort)(graph_ir)
    label_list = [graph_ir.label(node_id, config.strictness) for node_id in order]

    def accept(instance_node_list: list) -> object:
        """Run P3 on one candidate, plus the nested-pattern rule."""
        pattern = verify(graph_ir, instance_node_list, graph_output_node, reject_counter)
        if pattern is None or pattern.gain <= 0:
            return None
        if b_nested and _is_homogeneous_run(gs_node_list, instance_node_list[0]):
            reject_counter["homogeneous_run"] = reject_counter.get("homogeneous_run", 0) + 1
            return None
        return pattern

    def enrich(pattern, committed_node: set):
        """Look for occurrences of an accepted pattern that the search did not report.

        Growth fixes a block that came back *truncated*; this fixes one that came
        back *one instance short*, which the stress test shows is the more common
        of the two.
        """
        if pattern is None or config.method == "serial":
            return pattern
        node_list = [instance.node_id_list for instance in pattern.instance_list]
        extra = find_more_instances(graph_ir, order, config.strictness, node_list, committed_node)
        if not extra:
            return pattern
        position = {node_id: i for i, node_id in enumerate(order)}
        merged = sorted(node_list + extra, key=lambda n: position[n[0]])
        bigger = accept(merged)
        return bigger if bigger is not None and bigger.gain > pattern.gain else pattern

    def accept_with_backoff(member: list):
        """Verify a grown group, shrinking a node at a time if it overshot."""
        for size in range(len(member[0]), config.min_size - 1, -1):
            pattern = accept([node_list[:size] for node_list in member])
            if pattern is not None:
                return pattern
        return None

    def best_serial(committed_node: set, n_want: int) -> list:
        """The `n_want` highest-gain surviving candidates of the 1-D reduction.

        Each candidate is also offered to lockstep growth first: the 1-D search
        rarely misses a block outright, it reports it truncated or split because
        a foreign node landed in the middle of the run, and growth from the
        partial answer recovers the rest.
        """
        masked = [("<taken>", position) if node_id in committed_node else label_list[position] for position, node_id in enumerate(order)]
        result = []
        for candidate in find_candidates(masked, config.min_repeat, config.min_size):
            if candidate.gain <= 0:
                break
            instance_node_list = [[order[p] for p in range(position, position + candidate.length)] for position in candidate.position_list]
            pattern = accept(instance_node_list)
            if pattern is None:
                continue
            if config.method in ["parallel", "auto"]:
                member = extend_group(graph_ir, order, config.strictness, instance_node_list, config.max_block_node, committed_node)
                grown = accept_with_backoff(member) if member is not None else None
                if grown is not None and grown.gain > pattern.gain:
                    pattern = grown
            result.append(enrich(pattern, committed_node))
            if len(result) >= n_want:
                break
        return result

    def best_parallel(committed_node: set, n_want: int) -> list:
        """The `n_want` highest-gain candidates grown from scratch, for blocks the 1-D search never saw."""
        result = []
        for member in find_parallel_candidates(graph_ir, order, config.strictness, config.wl_radius, config.min_repeat, config.min_size, config.max_block_node, committed_node):
            pattern = accept_with_backoff(member)
            if pattern is not None:
                result.append(enrich(pattern, committed_node))
                if len(result) >= n_want:
                    break
        return result

    # Both searches compete inside ONE greedy loop, scored by the same MDL gain.
    # Running them in sequence instead would let the 1-D search commit a truncated
    # block first, leaving the graph-space search nothing to improve on.
    source_list = []
    if config.method in ["serial", "auto"]:
        source_list.append(best_serial)
    if config.method in ["parallel", "auto"]:
        source_list.append(best_parallel)

    # Beam search over "which pattern to commit next", scored by total MDL gain.
    # `beam == 1` is plain greedy. A wider beam lets the search keep the runner-up
    # instead, which matters when the best-looking candidate happens to eat the
    # nodes a much better pattern needed.
    beam = [(0, frozenset(), ())]
    for _ in range(config.max_pattern):
        nxt, b_progressed = {}, False
        for state in beam:
            total, committed_node, chosen = state
            successor = [pattern for source in source_list for pattern in source(set(committed_node), config.beam)]
            if not successor:
                # A state with nothing left to commit is a finished solution. It has
                # to stay in the beam: dropping it lets a worse but still extendable
                # state win, which made a wide beam score *below* plain greedy.
                nxt.setdefault(committed_node, state)
                continue
            b_progressed = True
            for pattern in successor:
                node_set = frozenset(node_id for instance in pattern.instance_list for node_id in instance.node_id_list)
                key = committed_node | node_set
                if key in nxt and nxt[key][0] >= total + pattern.gain:
                    continue
                nxt[key] = (total + pattern.gain, key, chosen + (pattern, ))
        if not b_progressed:
            break
        beam = sorted(nxt.values(), key=lambda state: -state[0])[:config.beam]

    return list(max(beam, key=lambda state: state[0])[2]), gs_node_list, len(order)

def _fold_levels(graph: gs.Graph, config: OutlineConfig, report: dict, root: gs.Graph | None = None, name_prefix: str = "Block", n_block_start: int = 0) -> int:
    """Fold repeatedly: level 0 folds nodes into blocks, level 1 folds repeated
    *groups of blocks* into outer functions, and so on up to `max_level`.

    Because the selection in P4 is gain driven, it always grabs the biggest
    win first. A second level therefore only finds anything when the inner
    block is *more frequent* than the group around it -- e.g. 6 layers arranged
    as 2 encoders of 3. On a plain chain of identical blocks level 0 already
    took the whole repetition and level 1 correctly finds nothing.
    """
    # Call node name -> the ORIGINAL node names it stands for, resolved through
    # every level, so the report can always point back at the flat graph.
    root = root if root is not None else graph
    origin = {}
    n_function, n_block = 0, n_block_start
    report.setdefault("levels", [])
    report.setdefault("patterns", [])
    reject_counter = report.setdefault("rejected", {})

    for level in range(config.max_level):
        pattern_list, gs_node_list, n_node_before = _find_patterns(graph, config, reject_counter, b_nested=level > 0)
        if not pattern_list:
            report["levels"].append({"level": level, "n_node_before": n_node_before, "n_node_after": n_node_before, "patterns": []})
            break
        if level == 0:
            report.setdefault("n_node_mined", n_node_before)

        level_pattern = []
        for pattern in pattern_list:
            name = f"{name_prefix}{n_block}"
            n_block += 1
            instance_origin = []
            for instance in pattern.instance_list:
                covered = []
                for node_id in instance.node_id_list:
                    node_name = gs_node_list[node_id].name
                    covered.extend(origin.get(node_name, [node_name]))
                instance_origin.append(covered)
            backend, loop_stat, loop_reason = "function", None, ""
            if config.backend == "loop":
                plan_list, loop_reason = analyse_loop(gs_node_list, pattern)
                if plan_list:
                    backend = "loop"
                    loop_stat = [outline_pattern_as_loop(graph, gs_node_list, plan, f"{name}_chain{c}" if len(plan_list) > 1 else name, root) for c, plan in enumerate(plan_list)]
            if backend == "function":
                outline_pattern(graph, gs_node_list, pattern, name, config.domain, root)
                n_function += 1
                for k, covered in enumerate(instance_origin):
                    origin[f"{name}_call_{k}"] = covered
            else:
                # One `Loop` node stands for the whole chain, so the whole chain maps
                # back to that single node.
                origin[name] = [node_name for covered in instance_origin for node_name in covered]

            entry = {
                "name": name,
                "level": level,
                "backend": backend,
                "loop_rejected_because": loop_reason,
                "loop": loop_stat,
                "domain": config.domain,
                "size": pattern.length,
                "n_instance": len(pattern.instance_list),
                "gain": pattern.gain,
                "n_function_input": len(pattern.instance_list[0].external_input),
                "n_function_output": len(pattern.instance_list[0].external_output),
                # How many *original* nodes one instance stands for, after resolving nesting
                "n_original_node": len(instance_origin[0]),
                "instances": instance_origin,
            }
            level_pattern.append(entry)
            report["patterns"].append(entry)

        graph.cleanup()
        sort_nodes_in_place(graph)
        report["levels"].append({
            "level": level,
            "n_node_before": n_node_before,
            "n_node_after": len(graph.nodes),
            "patterns": [p["name"] for p in level_pattern],
        })

    report["n_function"] = report.get("n_function", 0) + n_function
    report["n_loop"] = sum(1 for p in report["patterns"] if p["backend"] == "loop")
    return n_block

def _cross_check(original_file: Path, new_file: Path, seed: int, tolerance: float) -> dict:
    """Feed both models the same random inputs through onnxruntime and compare."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {"status": "skipped", "reason": "onnxruntime not installed"}

    try:
        session_a = ort.InferenceSession(str(original_file), providers=["CPUExecutionProvider"])
        session_b = ort.InferenceSession(str(new_file), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(seed)
        feed = {}
        dtype_map = {"tensor(float)": np.float32, "tensor(double)": np.float64, "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(bool)": np.bool_}
        for meta in session_a.get_inputs():
            shape = [d if isinstance(d, int) else 1 for d in meta.shape]
            dtype = dtype_map.get(meta.type)
            if dtype is None:
                return {"status": "skipped", "reason": f"unsupported input type {meta.type}"}
            feed[meta.name] = (rng.random(shape).astype(dtype) - 0.5) if dtype in [np.float32, np.float64] else np.ones(shape, dtype=dtype)
        out_a = session_a.run(None, feed)
        out_b = session_b.run(None, feed)
        # `equal_nan` matters: a model may legitimately produce NaN (`Sqrt` of a
        # negative input, an overflowing `Exp`), and `nan != nan` would then
        # report a mismatch for two byte-identical outputs.
        b_equal = all(np.array_equal(a, b, equal_nan=True) for a, b in zip(out_a, out_b))
        # Relative error is taken **per output**, not against the largest output of
        # the model: a model with one big and one tiny output would otherwise hide
        # an arbitrarily large error on the tiny one.
        diff, relative = 0.0, 0.0
        for a, b in zip(out_a, out_b):
            with np.errstate(invalid="ignore"):  # inf - inf is expected here, not an error
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            if not np.isfinite(d).any():
                continue
            this = float(np.nanmax(d))
            scale = float(np.nanmax(np.abs(a))) if np.isfinite(a).any() else 0.0
            diff = max(diff, this)
            relative = max(relative, this / scale if scale else this)
        b_ok = b_equal or (tolerance > 0 and relative <= tolerance)
        return {"status": "pass" if b_ok else "mismatch", "max_abs_diff": diff, "max_rel_diff": relative, "bit_exact": b_equal}
    except Exception as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}

def _tensorrt_check(onnx_file: Path) -> dict:
    """Can TensorRT still import the outlined model? Reported, never gating in M1.

    TensorRT inlines local functions, so a successful parse also tells us the
    layer count is unchanged, i.e. outlining costs nothing downstream.

    `TRTWrapperV1` already owns the builder / network / logger triple, so this
    only has to add the parser. Imported lazily: the outliner is useful without
    TensorRT installed, and `tensorrt_cookbook` imports this module at load time.
    """
    try:
        import tensorrt as trt

        from ..utils_class import TRTWrapperV1
    except ImportError:
        return {"status": "skipped", "reason": "tensorrt not installed"}
    try:
        tw = TRTWrapperV1(logger=trt.Logger.Severity.ERROR)
        parser = trt.OnnxParser(tw.network, tw.logger)
        if not parser.parse_from_file(str(onnx_file)):
            return {"status": "fail", "reason": str(parser.get_error(0)) if parser.num_errors else "unknown"}
        return {"status": "pass", "n_layer": tw.network.num_layers}
    except Exception as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}

def outline(input_file, output_file, config: OutlineConfig | None = None) -> dict:
    """Outline the repeated sub-graphs of `input_file` into `output_file`.

    Returns the report dict, which is also what the CLI dumps as JSON.
    """
    config = config or OutlineConfig()
    input_file, output_file = Path(input_file), Path(output_file)
    report = {"input": str(input_file), "output": str(output_file), "config": vars(config).copy()}

    model = onnx.load(input_file)
    report["n_node_input"] = len(model.graph.node)
    if config.preprocess:
        from .preprocess import preprocess
        model, report["preprocess"] = preprocess(model)
    else:
        report["preprocess"] = {"tool": "disabled", "n_node_before": len(model.graph.node), "n_node_after": len(model.graph.node)}

    graph = gs.import_onnx(model)
    graph.toposort()
    # The input may already hold local functions -- `torch.onnx.export` writes them
    # when `export_modules_as_functions` is used, so a partly outlined model is a
    # perfectly normal input. They are carried through untouched (their bodies are
    # not mined) and have to be excluded from the one-function-per-pattern check.
    report["n_function_preexisting"] = len(graph.functions)
    # Note which `Loop` / `If` bodies were there to begin with. Folding the main
    # graph both *creates* sub-graphs (the `Loop` back end) and can detach ones
    # that were absorbed into a function, so the list is taken again afterwards
    # and intersected with this one.
    original_subgraph = {id(sg) for sg in graph.subgraphs(recursive=True)} if config.subgraph else set()

    report["n_node_mined"] = len(graph.nodes)
    n_block = _fold_levels(graph, config, report)

    subgraph_list = [sg for sg in graph.subgraphs(recursive=True) if id(sg) in original_subgraph] if config.subgraph else []
    report["subgraph"] = {"n_subgraph": len(subgraph_list), "n_node": sum(len(sg.nodes) for sg in subgraph_list), "mined": config.subgraph}
    for k, subgraph in enumerate(subgraph_list):
        # A `FunctionProto` belongs to the model, not to a graph, so the functions
        # a sub-graph produces are registered on the root and called from inside
        # the body. ONNX allows exactly that.
        n_block = _fold_levels(subgraph, config, report, root=graph, name_prefix=f"Sub{k}Block", n_block_start=0)
        subgraph.cleanup()
        sort_nodes_in_place(subgraph)

    graph.cleanup()
    sort_nodes_in_place(graph)
    model_new = gs.export_onnx(graph)
    report["n_node_output"] = len(model_new.graph.node)
    assert report["n_function"] + report["n_function_preexisting"] == len(model_new.functions), "one FunctionProto per pattern is a hard invariant"
    # Coverage counts *original* nodes, so nesting cannot inflate it past 100%.
    # Sub-graph bodies are part of what was mined, so they belong in the denominator:
    # a model whose main graph is a single `Loop` would otherwise report 1200%.
    covered = {node_name for p in report["patterns"] for instance in p["instances"] for node_name in instance}
    n_mined = report["n_node_mined"] + report["subgraph"]["n_node"]
    report["coverage"] = len(covered) / n_mined if n_mined else 0.0

    # Nothing is written to disk until the model is known to be well formed:
    # a half-baked output file is worse than no output file.
    verification = {}
    report["verification"] = verification
    try:
        onnx.checker.check_model(model_new)
        verification["onnx_checker"] = "pass"
    except Exception as e:
        verification["onnx_checker"] = f"{type(e).__name__}: {e}"
        raise RuntimeError(f"outlined model failed onnx.checker, nothing written: {e}") from e

    onnx.save(model_new, output_file)

    # Compare against the *pre-processed* model, so any difference can only come
    # from the outlining itself and not from constant folding.
    reference_file = output_file.with_suffix(".reference.onnx")
    onnx.save(model, reference_file)
    verification["onnxruntime"] = _cross_check(reference_file, output_file, config.seed, config.tolerance)
    reference_file.unlink(missing_ok=True)
    verification["tensorrt"] = _tensorrt_check(output_file)

    if config.verify == "strict" and verification["onnxruntime"].get("status") != "pass":
        output_file.unlink(missing_ok=True)
        raise RuntimeError(f"outlined model failed the onnxruntime cross check: {verification['onnxruntime']}")

    return report
