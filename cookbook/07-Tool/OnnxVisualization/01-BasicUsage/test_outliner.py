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
"""Regression tests for the outliner.

Every model comes from `../00-ModelZoo/model_zoo.py`, which also holds the
expected answer for each one in its `EXPECT` table. Nothing is built here, so a
test model and its documented behaviour cannot drift apart.

The suite has two halves:

* the **zoo sweep**, one case per entry of `EXPECT`, which checks the pattern
  list, `onnx.checker`, the onnxruntime cross check and the function invariants;
* the **deep cases**, which take a few of those models further -- nesting levels,
  the `Loop` back end and its rejection reasons, sub-graph mining on and off,
  beam width monotonicity, and the two search methods on a planted block.

Three of the models exist purely because an arbitrary topological order gets them
wrong, and they are the reason the canonical order exists:

* `serial_plus_parallel` -- an arbitrary order misses the pattern entirely;
* `internal_branch`      -- parallelism inside a block must stay harmless;
* `two_tower`            -- an arbitrary order reports a *false* pattern that
  interleaves two unrelated towers.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import onnx

from tensorrt_cookbook import OutlineConfig, outline

sys.path.insert(0, str(Path(__file__).parent.parent / "00-ModelZoo"))
import model_zoo
from model_zoo import EXPECT

# ================================================================ Test driver

RESULT = []

def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one assertion."""
    RESULT.append((name, condition, detail))
    print(f"    [{'PASS' if condition else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return

def run_case(title: str, model_or_file, expect_pattern: list, work_path: Path, *, preprocess: bool = True, min_size: int = 3, max_level: int = 1, method: str = "auto", backend: str = "function", subgraph: bool = True, expect_level: list | None = None) -> dict:
    """Outline one model and check the patterns against `expect_pattern`.

    `expect_pattern` is a list of `(size, n_instance)` sorted the same way the
    outliner reports them (by decreasing gain). `expect_level`, when given, is
    the per-pattern nesting level in the same order.
    """
    print(f"=== {title}")
    if isinstance(model_or_file, onnx.ModelProto):
        input_file = work_path / f"{title}.onnx"
        onnx.save(model_or_file, input_file)
    else:
        input_file = Path(model_or_file)
    output_file = work_path / f"{title}-outlined.onnx"

    report = outline(input_file, output_file, OutlineConfig(preprocess=preprocess, min_size=min_size, max_level=max_level, method=method, backend=backend, subgraph=subgraph))
    found = [(p["size"], p["n_instance"]) for p in report["patterns"]]
    print(f"    {report['n_node_input']} nodes -> mine {report['n_node_mined']} -> main graph {report['n_node_output']}"
          f" + {report['n_function']} function, coverage {report['coverage']:.1%}")

    check(f"{title}: patterns", found == expect_pattern, f"expected {expect_pattern}, got {found}")
    if expect_level is not None:
        level = [p["level"] for p in report["patterns"]]
        check(f"{title}: nesting levels", level == expect_level, f"expected {expect_level}, got {level}")
    check(f"{title}: onnx.checker", report["verification"]["onnx_checker"] == "pass", str(report["verification"]["onnx_checker"]))
    check(f"{title}: onnxruntime bit-exact", report["verification"]["onnxruntime"].get("status") == "pass", str(report["verification"]["onnxruntime"]))
    check(f"{title}: one function per pattern", report["n_function"] == len(report["patterns"]), f"{report['n_function']} functions for {len(report['patterns'])} patterns")

    model_new = onnx.load(output_file)
    call_op = {(n.domain, n.op_type) for n in model_new.graph.node if n.domain}
    declared = {(f.domain, f.name) for f in model_new.functions}
    check(f"{title}: every call resolves to a function", call_op <= declared, f"{call_op} vs {declared}")
    return report

def run_loop_case(title: str, model_or_file, work_path: Path, *, expect_loop: int, expect_reason: str = "", preprocess: bool = True) -> dict:
    """Outline with the `Loop` back end and check what came out."""
    print(f"=== {title}")
    if isinstance(model_or_file, onnx.ModelProto):
        input_file = work_path / f"{title}.onnx"
        onnx.save(model_or_file, input_file)
    else:
        input_file = Path(model_or_file)
    output_file = work_path / f"{title}-loop.onnx"
    report = outline(input_file, output_file, OutlineConfig(preprocess=preprocess, backend="loop"))
    reason = "; ".join(p["loop_rejected_because"] for p in report["patterns"] if p["loop_rejected_because"])
    print(f"    main graph {report['n_node_output']} nodes, {report['n_function']} function(s), {report['n_loop']} loop pattern(s)"
          f"{', rejected: ' + reason if reason else ''}")

    check(f"{title}: loop patterns", report["n_loop"] == expect_loop, f"expected {expect_loop}, got {report['n_loop']}")
    check(f"{title}: onnx.checker", report["verification"]["onnx_checker"] == "pass", str(report["verification"]["onnx_checker"]))
    check(f"{title}: onnxruntime within tolerance", report["verification"]["onnxruntime"].get("status") == "pass", str(report["verification"]["onnxruntime"]))
    check(f"{title}: relative error is fp32 noise", report["verification"]["onnxruntime"].get("max_rel_diff", 1.0) < 1e-5, str(report["verification"]["onnxruntime"].get("max_rel_diff")))
    check(f"{title}: tensorrt parses", report["verification"]["tensorrt"].get("status") in ["pass", "skipped"], str(report["verification"]["tensorrt"]))
    if expect_reason:
        check(f"{title}: rejection reason", expect_reason in reason, f"expected {expect_reason!r} in {reason!r}")
    return report

def sweep_zoo(work_path: Path) -> None:
    """One case per entry of `model_zoo.EXPECT`, in the order the zoo lists them."""
    print("\n" + "#" * 30 + f" zoo sweep: {len(EXPECT)} models")
    for name in model_zoo.name_list():
        expect = EXPECT.get(name)
        if expect is None:
            check(f"{name}: has an expected answer in model_zoo.EXPECT", False, "add one, or the model is not a test case")
            continue
        print(f"--- {expect['note']}")
        run_case(name, model_zoo.ensure(name), expect["pattern"], work_path, **expect["config"])
    return

def deep_cases(work_path: Path) -> None:
    """The cases that take one zoo model further than a single outline call."""
    print("\n" + "#" * 30 + " deep cases")

    # Nesting. `max_level=1` must behave exactly like before the feature existed.
    nested = model_zoo.ensure("nested_two_level")
    run_case("nested_level1", nested, [(3, 6)], work_path, preprocess=False, max_level=1, expect_level=[0])
    run_case("nested_level2", nested, [(3, 6), (4, 2)], work_path, preprocess=False, max_level=2, expect_level=[0, 1])
    run_case("nested_level5", nested, [(3, 6), (4, 2)], work_path, preprocess=False, max_level=5, expect_level=[0, 1])

    # A plain chain has nothing left to nest: level 0 already took the whole repetition
    transformer = model_zoo.ensure("transformer_6layer")
    run_case("transformer_level3", transformer, [(41, 6)], work_path, max_level=3, expect_level=[0])

    # The `Loop` back end: the only folding that shrinks the TensorRT network.
    report = run_loop_case("loop_transformer", transformer, work_path, expect_loop=1)
    check("loop_transformer: main graph is a single Loop", report["n_node_output"] == 1, f"{report['n_node_output']} nodes")
    check("loop_transformer: TensorRT layers drop", report["verification"]["tensorrt"].get("n_layer", 10 ** 9) < 200, f"{report['verification']['tensorrt'].get('n_layer')} layers, the function version has 389")
    # Two chains of three, separated by a tanh -> one Loop each
    run_loop_case("loop_two_chain", model_zoo.ensure("transformer_two_stage"), work_path, expect_loop=1)
    # Multiple outputs: one loop variable plus a scan output sliced back per iteration
    report = run_loop_case("loop_scan_output", model_zoo.ensure("loop_scan_output"), work_path, expect_loop=1, preprocess=False)
    stat = report["patterns"][0]["loop"][0]
    check("loop_scan_output: one carried, one scan", (stat["n_carried"], stat["n_scan_output"]) == (1, 1), str(stat))
    check("loop_scan_output: slices handed back", stat["n_slice_node"] == 4, str(stat))
    # Two values handed to the next iteration: two loop-carried variables
    report = run_loop_case("loop_two_carried", model_zoo.ensure("loop_two_carried"), work_path, expect_loop=1, preprocess=False)
    stat = report["patterns"][0]["loop"][0]
    check("loop_two_carried: two loop variables", stat["n_carried"] == 2, str(stat))
    # Two towers are two chains, so both really do fold into loops
    run_loop_case("loop_two_tower", model_zoo.ensure("two_tower"), work_path, expect_loop=2, preprocess=False)
    # A fan-out is not a chain at all, so it must fall back to a function
    run_loop_case("loop_fan_out", model_zoo.ensure("fan_out"), work_path, expect_loop=0, expect_reason="chains", preprocess=False)

    # Mining inside `Loop` / `If` bodies. The main graph has one node, so a tool
    # that only looks at the top level finds nothing.
    for title, name, n_expect in [("subgraph_loop", "loop_with_repeats", 1), ("subgraph_if", "if_with_repeats", 2)]:
        report = run_case(title, model_zoo.ensure(name), [(3, 4)] * n_expect, work_path, preprocess=False)
        check(f"{title}: main graph untouched", report["n_node_output"] == 1, f"{report['n_node_output']} nodes")
        check(f"{title}: sub-graph was seen", report["subgraph"]["n_subgraph"] == n_expect, str(report["subgraph"]))
        model_new = onnx.load(work_path / f"{title}-outlined.onnx")
        called = [(n.domain, n.op_type) for node in model_new.graph.node for a in node.attribute if a.type == onnx.AttributeProto.GRAPH for n in a.g.node if n.domain]
        check(f"{title}: body calls the function", len(called) == 4 * n_expect, f"{len(called)} call nodes: {set(called)}")
    # Turning it off must go back to leaving them alone
    report = run_case("subgraph_off", model_zoo.ensure("loop_with_repeats"), [], work_path, preprocess=False, subgraph=False)
    check("subgraph_off: nothing folded", report["n_function"] == 0, str(report["n_function"]))

    # A model PyTorch already outlined: its function has to survive untouched, and
    # the one-function-per-pattern invariant has to account for it.
    report = run_case("preexisting_function", model_zoo.ensure("flat_mlp_as_function"), [], work_path)
    check("preexisting_function: carried through", report["n_function_preexisting"] == 1, str(report["n_function_preexisting"]))
    model_new = onnx.load(work_path / "preexisting_function-outlined.onnx")
    check("preexisting_function: still called 4 times", sum(1 for n in model_new.graph.node if n.domain) == 4, str([(n.domain, n.op_type) for n in model_new.graph.node]))

    # Shape-sensitive operators. The premise "node label = (op_type, all
    # attributes)" is what keeps blocks that disagree on `Transpose.perm` or
    # `Concat.axis` apart. `--strictness L0` drops attributes from the label, so
    # it is the direct experiment on that premise: it *must* merge them, and the
    # numeric gate *must* be the thing that notices.
    for name, expect_l1 in [("transpose_perm", [(3, 2), (3, 2)]), ("concat_axis", [(3, 2), (3, 2)])]:
        onnx_file = model_zoo.ensure(name)
        run_case(f"{name}_L1", onnx_file, expect_l1, work_path, preprocess=False)

        report = outline(onnx_file, work_path / f"{name}-L0.onnx", OutlineConfig(preprocess=False, strictness="L0"))
        found = [(p["size"], p["n_instance"]) for p in report["patterns"]]
        verification = report["verification"]
        check(f"{name}_L0: attributes ignored, so the blocks wrongly merge", found == [(3, 4)], str(found))
        # Both structural gates wave it through -- this is the whole lesson
        check(f"{name}_L0: onnx.checker still passes", verification["onnx_checker"] == "pass", str(verification["onnx_checker"]))
        check(f"{name}_L0: TensorRT still parses", verification["tensorrt"].get("status") in ["pass", "skipped"], str(verification["tensorrt"]))
        check(f"{name}_L0: only the numeric check catches it", verification["onnxruntime"].get("status") != "pass", str(verification["onnxruntime"])[:90])

    # The mirror case: `Reshape` takes its target shape as an *input*, which is
    # passed per call site, so four different targets must still be one pattern.
    # This guards the other direction -- a label that also split these would be
    # too strict and would lose real compression.
    report = run_case("reshape_shape_input", model_zoo.ensure("reshape_shape_input"), [(4, 4)], work_path, preprocess=False)
    check("reshape_shape_input: the shapes became function inputs", report["patterns"][0]["n_function_input"] >= 3, f"{report['patterns'][0]['n_function_input']} inputs")

    # Beam search must never score below plain greedy: a wider beam contains
    # greedy's own path. It did score below, because a state with nothing left to
    # commit was dropped from the beam instead of kept as a finished answer.
    for name in ["nested_two_level", "two_tower", "shared_input"]:
        gain = {}
        for width in [1, 2, 4]:
            report = outline(model_zoo.ensure(name), work_path / f"beam_{name}-b{width}.onnx", OutlineConfig(preprocess=False, beam=width))
            gain[width] = sum(p["gain"] for p in report["patterns"])
        check(f"beam_{name}: wider beam never loses gain", gain[4] >= gain[2] >= gain[1], str(gain))

    # Path B. The planted 5-node block is broken up in the topological order, so
    # the 1-D search alone can only report 4 of its 5 nodes.
    split_file, planted = model_zoo.planted_block_with_answer(work_path / "split_block.onnx")
    for method, expect in [("serial", [(4, 2)]), ("auto", [(5, 2)])]:
        report = run_case(f"split_block_{method}", split_file, expect, work_path, preprocess=False, method=method)
        recovered = sum(1 for names in planted if any(set(names) <= set(i) for p in report["patterns"] for i in p["instances"]))
        check(f"split_block_{method}: planted instances recovered", recovered == (len(planted) if method == "auto" else 0), f"{recovered} of {len(planted)}")
    return

def main() -> int:
    """Run every case, return a process exit code."""
    work_path = Path(tempfile.mkdtemp(prefix="outliner-test-"))
    try:
        sweep_zoo(work_path)
        deep_cases(work_path)
    finally:
        shutil.rmtree(work_path, ignore_errors=True)

    n_fail = sum(1 for _, ok, _ in RESULT if not ok)
    print("\n" + "=" * 30 + f" {len(RESULT) - n_fail}/{len(RESULT)} checks passed")
    for name, ok, detail in RESULT:
        if not ok:
            print(f"    FAILED {name}: {detail}")
    return 1 if n_fail else 0

if __name__ == "__main__":
    code = main()
    print("\nFinish")
    sys.exit(code)
