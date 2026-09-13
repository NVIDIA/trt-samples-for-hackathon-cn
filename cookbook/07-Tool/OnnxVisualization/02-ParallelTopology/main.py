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
"""Does the outliner survive a genuinely branchy topology?

Everything tested so far was a chain of identical blocks. This builds five
structurally identical transformer encoder layers wired into a DAG with a fork,
a re-join and a skip connection:

    x --> A --+--> B ------+
              |            |--> h = B * C --+
              +--> C --+---+                +--> Add --> D --+
              |        |                                     |--> f = D + E
              +········+--> E ---------------------------->--+
              :
              +········(the A --> D skip, `case_with_skip` only)

Two readings of "A -> D" are both exercised:

* `case_with_skip`      -- A really feeds D, so D's input is `h + A_out`.
  This is the harder case: D's merge point differs from every other block's.
* `case_transitive_only` -- "A -> D" only meant "D is downstream of A", D eats h.

The interesting question is not whether the five blocks are found (they are),
but whether the canonical topological order keeps each of them contiguous when
two of them are siblings of a fork.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

output_path = Path(__file__).parent
# Both variants of the DAG live in the model zoo next door, together with the
# `BranchyNet` module that produces them and the pattern they are expected to
# yield -- the very same entries `../01-BasicUsage/test_outliner.py` asserts against.
sys.path.insert(0, str(output_path.parent / "00-ModelZoo"))

from model_zoo import ensure
from tensorrt_cookbook import case_mark, outline

np.random.seed(31193)

N_MODEL = 64
N_SEQ = 16
N_B = 2

VERDICT = []  # (case name, all checks passed)

ZOO_NAME = {"skip": "transformer_branchy", "transitive": "transformer_branchy_transitive"}

def describe(onnx_file: Path, report: dict) -> None:
    """Print what the outliner made of one variant."""
    print(f"    {report['n_node_input']:>5} nodes in the input file")
    print(f"    {report['preprocess']['n_node_after']:>5} nodes after {report['preprocess']['tool']}")
    print(f"    {report['n_node_output']:>5} nodes in the main graph + {report['n_function']} local function(s), coverage {report['coverage']:.1%}")
    for pattern in report["patterns"]:
        print(f"          {pattern['name']}: {pattern['size']:>3} nodes x {pattern['n_instance']} instances, "
              f"interface {pattern['n_function_input']} in / {pattern['n_function_output']} out")
    print(f"    rejected: {report['rejected'] or 'none'}")
    print(f"    verification: {report['verification']}")

    model = onnx.load(report["output"])
    print(f"    main graph: {[(n.domain or '-', n.op_type) for n in model.graph.node]}")
    return

def check_five_blocks(report: dict) -> bool:
    """One pattern covering all five encoder layers, and every gate green."""
    v = report["verification"]
    return (len(report["patterns"]) == 1 and report["patterns"][0]["n_instance"] == 5 and v["onnx_checker"] == "pass" and v["onnxruntime"].get("status") == "pass" and v["tensorrt"].get("status") == "pass")

@case_mark
def case_with_skip() -> bool:
    """`A -> D` read as a real edge: D eats `h + A_out`."""
    onnx_file = ensure(ZOO_NAME["skip"])
    report = outline(onnx_file, output_path / "model-branchy-skip-outlined.onnx")
    describe(onnx_file, report)
    (output_path / "report-skip.json").write_text(json.dumps(report, indent=2))
    ok = check_five_blocks(report)
    print(f"    -> all five blocks in one function, all gates green: {ok}")
    return ok

@case_mark
def case_transitive_only() -> bool:
    """`A -> D` read as "D is downstream of A": D eats `h`."""
    onnx_file = ensure(ZOO_NAME["transitive"])
    report = outline(onnx_file, output_path / "model-branchy-transitive-outlined.onnx")
    describe(onnx_file, report)
    (output_path / "report-transitive.json").write_text(json.dumps(report, indent=2))
    ok = check_five_blocks(report)
    print(f"    -> all five blocks in one function, all gates green: {ok}")
    return ok

@case_mark
def case_numeric_cross_check() -> None:
    """Belt and braces: run both variants through onnxruntime against the original."""
    data = {"x": np.random.rand(N_B, N_SEQ, N_MODEL).astype(np.float32) - 0.5}
    for tag in ["skip", "transitive"]:
        original = ensure(ZOO_NAME[tag])
        outlined = output_path / f"model-branchy-{tag}-outlined.onnx"
        if not outlined.exists():
            continue
        a = ort.InferenceSession(str(original), providers=["CPUExecutionProvider"]).run(None, data)[0]
        b = ort.InferenceSession(str(outlined), providers=["CPUExecutionProvider"]).run(None, data)[0]
        print(f"    {tag:<11}: max |original - outlined| = {np.max(np.abs(a - b)):.3e}")
    return

if __name__ == "__main__":
    VERDICT.append(("with_skip", case_with_skip()))
    VERDICT.append(("transitive_only", case_transitive_only()))
    case_numeric_cross_check()

    print("\n" + "=" * 30 + " Verdict")
    for name, ok in VERDICT:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}: five blocks folded into one function, "
              f"checker + onnxruntime + TensorRT all pass")
    print("\nFinish")
    sys.exit(0 if all(ok for _, ok in VERDICT) else 1)
