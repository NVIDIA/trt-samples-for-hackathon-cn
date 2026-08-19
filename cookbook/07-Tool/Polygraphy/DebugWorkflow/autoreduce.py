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
"""Make `polygraphy debug reduce` auditable: keep every iteration, not just the answer.

`polygraphy debug reduce` bisects a failing model down to a minimal failing subgraph and
writes **one** file: the final `reduced.onnx`. That is the right default and a bad debugging
experience, because the two questions you ask next cannot be answered from it:

+ *"Is the reduced model still failing for the original reason?"* -- reduction is driven by
  a pass/fail command, and any command that fails for a second reason (a typo in the shape,
  a missing file, an OOM) reduces beautifully to something irrelevant. This failure mode is
  silent: you get a small model and a plausible story.
+ *"How did it get here?"* -- which node was dropped at which step, and did the failure
  survive each drop.

Polygraphy already has the mechanism, it is just not the default: `--artifacts` tells it to
save named files per iteration, and `--art-dir` where to put them. This script drives a real
reduction with that turned on, then reads the replay journal back and prints the audit
trail, so the reduction becomes a record rather than an assertion.

A from-scratch reimplementation of the idea behind an internal, proprietary
`polygraphy_autoreduce.py`; no code was taken from it.
"""

import json
import subprocess
from collections import OrderedDict
from pathlib import Path

import onnx

from tensorrt_cookbook import case_mark, cookbook_path

output_path = Path(__file__).parent
onnx_file = cookbook_path("00-Data", "model", "model-unknown.onnx")
artifact_directory = output_path / "reduce_artifacts"
replay_file = output_path / "data-replay.json"
reduced_file = output_path / "model-reduced.onnx"

result = OrderedDict()

def describe(path: Path) -> str:
    """Node count and operator histogram of an ONNX file."""
    if not path.exists():
        return "missing"
    model = onnx.load(path)
    counter = OrderedDict()
    for node in model.graph.node:
        counter[node.op_type] = counter.get(node.op_type, 0) + 1
    return f"{len(model.graph.node)} nodes {dict(sorted(counter.items()))}"

@case_mark
def case_reduce_with_artifacts() -> None:
    """Run the reduction, keeping every intermediate model.

    `--artifacts polygraphy_debug.onnx` names the file `debug reduce` writes each iteration;
    `--art-dir` collects them. Without those two flags the intermediates are overwritten in
    place and only the final model survives.

    Note the option is `--save-debug-replay`, not `--save-replay`: the shorter name is what
    `debug build`/`debug repeat` documentation reads like, and passing it here fails with
    `Unrecognized Options` rather than a suggestion.
    """
    for stale in [artifact_directory, replay_file, reduced_file]:
        if stale.is_dir():
            subprocess.run(["rm", "-rf", str(stale)], check=True)
        elif stale.exists():
            stale.unlink()

    command = [
        "polygraphy",
        "debug",
        "reduce",
        str(onnx_file),
        "--output",
        str(reduced_file),
        "--model-input-shapes",
        "inputT0:[1,1,28,28]",
        "--save-debug-replay",
        str(replay_file),
        "--artifacts",
        "polygraphy_debug.onnx",
        "--art-dir",
        str(artifact_directory),
        "--check",
        "polygraphy",
        "run",
        "polygraphy_debug.onnx",
        "--trt",
    ]
    process = subprocess.run(command, capture_output=True, text=True, cwd=str(output_path))
    print(f"    exit={process.returncode}")
    tail = [line for line in process.stdout.splitlines() if "Reduced model" in line or "PASSED" in line or "FAILED" in line]
    print(f"    {tail[-3:] if tail else process.stdout.splitlines()[-3:]}")
    result["returncode"] = process.returncode
    return

@case_mark
def case_audit_trail() -> None:
    """What survived: the per-iteration artifacts and the replay journal."""
    print(f"    original : {describe(onnx_file)}")
    print(f"    reduced  : {describe(reduced_file)}")

    saved = sorted(artifact_directory.rglob("*.onnx")) if artifact_directory.exists() else []
    print(f"    kept {len(saved)} intermediate model(s) under {artifact_directory.name}/")
    for path in saved[:10]:
        print(f"        {path.relative_to(output_path)}: {describe(path)}")
    result["n_artifact"] = len(saved)

    if replay_file.exists():
        replay = json.loads(replay_file.read_text())
        # The replay maps each tried configuration to its pass/fail outcome. It is the
        # record that answers "did the failure survive every step", which the final
        # `reduced.onnx` cannot.
        print(f"    replay journal: {len(replay)} top-level key(s) -> {list(replay)[:6]}")
        result["n_replay"] = len(replay)
    return

@case_mark
def case_why_it_matters() -> None:
    """A reduction driven by the *wrong* failure still succeeds, and looks the same.

    Here the check command is broken on purpose -- it names a tensor that does not exist --
    so every candidate 'fails'. `debug reduce` cannot tell that apart from the real bug and
    happily reduces to something meaningless. The only way to notice is to look at what the
    check actually reported, which is exactly what the artifacts and replay preserve.
    """
    bogus_output = output_path / "model-reduced-bogus.onnx"
    if bogus_output.exists():
        bogus_output.unlink()
    command = [
        "polygraphy",
        "debug",
        "reduce",
        str(onnx_file),
        "--output",
        str(bogus_output),
        "--model-input-shapes",
        "inputT0:[1,1,28,28]",
        "--check",
        "polygraphy",
        "run",
        "polygraphy_debug.onnx",
        "--trt",
        "--onnx-outputs",
        "this_tensor_does_not_exist",
    ]
    process = subprocess.run(command, capture_output=True, text=True, cwd=str(output_path))
    print(f"    reduction driven by a broken check: exit={process.returncode}")
    print(f"    produced a model anyway: {bogus_output.exists()}  ->  {describe(bogus_output)}")
    print(f"    real reduction was     : {describe(reduced_file)}")

    same = describe(bogus_output) == describe(reduced_file)
    print(f"    the two reduced models are indistinguishable: {same}")
    print("    On this 5-node model they even coincide, which IS the lesson: the artifact")
    print("    cannot tell you which failure drove the reduction. Only the recorded check")
    print("    output can, and that is what --artifacts / --save-debug-replay preserve:")
    reason = [line for line in process.stdout.splitlines() if "this_tensor_does_not_exist" in line or "Could not find" in line]
    print(f"        broken check reported: {reason[-1].split('] ')[-1][:110] if reason else '(see log)'}")
    result["bogus_exists"] = bogus_output.exists()
    result["same_as_real"] = same
    return

def main() -> None:
    case_reduce_with_artifacts()
    case_audit_trail()
    case_why_it_matters()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
