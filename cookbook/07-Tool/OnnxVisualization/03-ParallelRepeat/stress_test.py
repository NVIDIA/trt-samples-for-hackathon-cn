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
"""How often does the 1-D path actually miss a planted repeated block?

`DESIGN.md` assumed the topological-order reduction would be blind to *parallel*
repetition and that a second, graph-space search (path B) would be needed. Hand
written parallel structures did not reproduce that, so this measures it properly
instead of guessing: generate many random DAGs, plant K identical copies of a
random block in each, and count how often the outliner recovers all of them.

The theory behind the experiment:

* a block can only be outlined at all if it is **convex** (contracting it must
  not create a cycle), otherwise "compute part of it, leave, come back" cannot
  be one function call;
* if a set of disjoint node sets is convex, contracting them yields a DAG, so
  **some** topological order makes every one of them contiguous;
* therefore the 1-D reduction is not fundamentally blind to parallel structure.
  Its only weakness is picking the *wrong* order among the valid ones.

Blocks are planted convex by construction (their interior tensors are never
published), so every miss reported here is a real ordering failure.
"""

import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import onnx
import onnx.helper as oh

from tensorrt_cookbook import OutlineConfig, outline

N_CHANNEL = 4
OPSET = 17
FLOAT = onnx.TensorProto.FLOAT
UNARY = ["Relu", "Tanh", "Sigmoid", "Softplus", "Elu", "Selu", "Softsign", "Sqrt", "Exp", "Sin"]
BINARY = ["Add", "Mul", "Sub", "Min", "Max"]

def make_block_template(rng: random.Random, n_node: int, n_external_input: int) -> list:
    """A random connected mini-DAG: `[(op, [ref, ...]), ...]` in topological order.

    A ref is `("e", j)` for external input j or `("i", k)` for the output of the
    k-th node of the block.
    """
    template = []
    for k in range(n_node):
        pool = [("e", j) for j in range(n_external_input)] + [("i", j) for j in range(k)]
        if k > 0:  # Keep it connected: every node after the first touches the block
            must = [("i", j) for j in range(k)]
            first = rng.choice(must)
        else:
            first = rng.choice(pool)
        if rng.random() < 0.5:
            template.append((rng.choice(UNARY), [first]))
        else:
            template.append((rng.choice(BINARY), [first, rng.choice(pool)]))
    return template

def build_model(rng: random.Random, n_block_node: int, n_external_input: int, n_instance: int, n_filler: int, onnx_file: Path) -> tuple:
    """A random DAG with `n_instance` copies of one random block planted in it.

    Returns the file path and, per instance, the node names that belong to it.
    """
    template = make_block_template(rng, n_block_node, n_external_input)
    node_list, pool, instance_name = [], ["x"], []
    counter = 0

    def add_filler() -> None:
        """One random node wired from whatever is already available."""
        nonlocal counter
        op = rng.choice(UNARY + BINARY)
        n_input = 1 if op in UNARY else 2
        name = f"F{counter}"
        node_list.append(oh.make_node(op, [rng.choice(pool) for _ in range(n_input)], [f"f{counter}"], name))
        pool.append(f"f{counter}")
        counter += 1

    for i in range(n_instance):
        for _ in range(rng.randint(0, n_filler)):
            add_filler()
        # An instance's external inputs come from tensors that already exist, and only
        # its last node is published, so nothing outside can ever see its interior.
        # That makes every planted instance convex by construction.
        external = [rng.choice(pool) for _ in range(n_external_input)]
        internal, names = [], []
        for k, (op, ref_list) in enumerate(template):
            name = f"B{i}_{k}"
            input_list = [external[j] if kind == "e" else internal[j] for kind, j in ref_list]
            node_list.append(oh.make_node(op, input_list, [f"b{i}_{k}"], name))
            internal.append(f"b{i}_{k}")
            names.append(name)
        pool.append(internal[-1])
        instance_name.append(names)

    for _ in range(rng.randint(0, n_filler)):
        add_filler()

    consumed = {t for n in node_list for t in n.input}
    leaf = [t for t in pool if t not in consumed] or [pool[-1]]
    node_list.append(oh.make_node("Sum", leaf, ["y"], "Merge") if len(leaf) > 1 else oh.make_node("Identity", leaf, ["y"], "Merge"))

    graph = oh.make_graph(
        node_list,
        "planted",
        [oh.make_tensor_value_info("x", FLOAT, [N_CHANNEL])],
        [oh.make_tensor_value_info("y", FLOAT, [N_CHANNEL])],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, onnx_file)
    return onnx_file, instance_name

def classify(report: dict, instance_name: list) -> tuple:
    """How much of the planted structure came back?

    Returns a coarse verdict plus the fraction of planted *nodes* that ended up
    inside some reported instance. The fraction matters: "found 4 of the 5 nodes
    of every instance" is a very different outcome from "found nothing", and a
    binary verdict hides that.
    """
    found = [set(instance) for pattern in report["patterns"] for instance in pattern["instances"]]
    n_exact = sum(1 for names in instance_name if any(set(names) <= f for f in found))
    n_planted_node = sum(len(names) for names in instance_name)
    n_covered_node = sum(len(set(names) & f) for names in instance_name for f in found)
    fraction = n_covered_node / n_planted_node if n_planted_node else 0.0
    if n_exact == len(instance_name):
        verdict = "exact"
    elif n_exact > 0:
        verdict = "partial_mixed"
    elif n_covered_node:
        verdict = "partial"
    else:
        verdict = "missed"
    return verdict, fraction

def main() -> int:
    """Run the sweep and print a summary table."""
    n_case = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    work_path = Path(tempfile.mkdtemp(prefix="outliner-stress-"))
    verdict = Counter()
    fraction_list = []
    failure = []

    for case in range(n_case):
        rng = random.Random(31193 + case)
        n_block_node = rng.randint(3, 8)
        n_external_input = rng.randint(1, 3)
        n_instance = rng.randint(2, 6)
        n_filler = rng.randint(0, 4)
        onnx_file, instance_name = build_model(rng, n_block_node, n_external_input, n_instance, n_filler, work_path / f"case{case}.onnx")
        try:
            report = outline(onnx_file, work_path / f"case{case}-out.onnx", OutlineConfig(preprocess=False, min_size=3))
        except Exception as e:
            verdict["error"] += 1
            failure.append((case, "error", f"{type(e).__name__}: {e}"))
            continue

        result, fraction = classify(report, instance_name)
        verdict[result] += 1
        fraction_list.append(fraction)
        if report["verification"]["onnxruntime"].get("status") != "pass":
            verdict["numeric_mismatch"] += 1
            failure.append((case, "numeric", str(report["verification"]["onnxruntime"])))
        if result != "exact":
            failure.append((case, result, f"block={n_block_node}n/{n_external_input}in x{n_instance}, filler<={n_filler}, "
                            f"recovered {fraction:.0%} of the planted nodes, "
                            f"got {[(p['size'], p['n_instance']) for p in report['patterns']]}"))
        onnx_file.unlink(missing_ok=True)
        (work_path / f"case{case}-out.onnx").unlink(missing_ok=True)

    mean_fraction = sum(fraction_list) / len(fraction_list) if fraction_list else 0.0
    print("=" * 30 + f" {n_case} random planted-block models")
    print(f"    every instance recovered exactly : {verdict['exact']:>4}  ({verdict['exact'] / n_case:.1%})")
    print(f"    some exact, some partial         : {verdict['partial_mixed']:>4}")
    print(f"    all instances only partial       : {verdict['partial']:>4}")
    print(f"    planted block not touched at all : {verdict['missed']:>4}")
    print(f"    outliner raised                  : {verdict['error']:>4}")
    print(f"    numeric mismatch                 : {verdict['numeric_mismatch']:>4}   <- must be 0")
    print(f"    mean fraction of planted nodes recovered : {mean_fraction:.1%}")
    if failure:
        print("\n    first failures:")
        for case, kind, detail in failure[:10]:
            print(f"       case {case:>4} [{kind}] {detail}")

    import shutil
    shutil.rmtree(work_path, ignore_errors=True)
    # Only numeric errors are a bug; a missed pattern is a quality issue we want to measure.
    return 1 if verdict["numeric_mismatch"] or verdict["error"] else 0

if __name__ == "__main__":
    code = main()
    print("\nFinish")
    sys.exit(code)
