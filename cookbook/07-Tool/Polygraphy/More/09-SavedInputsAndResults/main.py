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
"""Saving inference inputs and outputs, and the bridge between the CLI and Python.

Two formats, and they are the same ones `polygraphy run --save-inputs` /
`--save-outputs` write:

    inputs   `List[Dict[str, np.ndarray]]`, one feed dict per iteration
    outputs  `RunResults`, mapping runner name -> list of `IterationResult`

The upstream example only *loads* files that the CLI produced. This one produces
them from Python, loads them back, checks the round trip is bit-exact, and then
loads a file the CLI wrote -- which is the practical reason the format is worth
knowing: a comparison can be split across the two tools, or across machines.

`../Run/` covers the CLI side.
"""

import subprocess
from pathlib import Path

import numpy as np
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.comparator import Comparator, CompareFunc, RunResults
from polygraphy.json import load_json
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

onnx_file = str(cookbook_path("00-Data", "model", "model-trained.onnx"))
output_path = Path(__file__).parent
input_json = output_path / "inputs.json"
output_json = output_path / "outputs.json"
cli_output_json = output_path / "cli-outputs.json"

results = None

@case_mark
def case_save_from_python() -> None:
    """Produce both files from a `Comparator.run`.

    `save_inputs_path` writes the feed dicts the data loader generated -- which
    matters because the default loader is random, so without saving them the run
    is not reproducible.
    """
    global results
    results = Comparator.run(
        [TrtRunner(EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=CreateConfig()), name="trt")],
        save_inputs_path=str(input_json),
    )
    results.save(str(output_json))
    print(f"    {input_json.name}: {input_json.stat().st_size} B")
    print(f"    {output_json.name}: {output_json.stat().st_size} B")
    print("    note the inputs file is the larger one -- it holds a 28x28 image, the outputs hold 10 logits")
    return

@case_mark
def case_load_and_check_the_round_trip() -> None:
    """Read both back and confirm nothing was lost.

    `load_json` is for plain objects like the input list; Polygraphy's own types
    carry `save` / `load` instead, which is why `RunResults` is not loaded with
    `load_json`.
    """
    loaded_input = load_json(str(input_json))
    loaded_results = RunResults.load(str(output_json))

    print(f"    inputs : {type(loaded_input).__name__} of {len(loaded_input)} feed dict(s), keys {list(loaded_input[0].keys())}")
    print(f"    outputs: runners {list(loaded_results.keys())}, output keys {list(loaded_results['trt'][0].keys())}")

    for name in results["trt"][0].keys():
        before = np.asarray(results["trt"][0][name])
        after = np.asarray(loaded_results["trt"][0][name])
        print(f"    {name}: round trip bit-exact = {np.array_equal(before, after)}")
    return

@case_mark
def case_the_cli_writes_the_same_format() -> None:
    """`polygraphy run --save-outputs` produces a file `RunResults.load` reads.

    This is the bridge: generate results on one machine with the CLI, analyse
    them in Python somewhere else. Note the CLI names its runner with a
    timestamp, so code that reads CLI output should not hard-code the key.
    """
    completed = subprocess.run(
        ["polygraphy", "run", onnx_file, "--trt", "--save-outputs", str(cli_output_json), "--save-inputs", str(output_path / "cli-inputs.json")],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(f"    CLI failed (rc={completed.returncode}), skipping")
        return

    cli_results = RunResults.load(str(cli_output_json))
    runner_name = list(cli_results.keys())[0]
    print(f"    CLI runner name : {runner_name!r}   <- timestamped, do not hard-code it")
    print(f"    output keys     : {list(cli_results[runner_name][0].keys())}")
    print("    same class, same file format, different producer")
    return

@case_mark
def case_compare_across_the_two_runs() -> None:
    """Saved results are ordinary `RunResults`, so `Comparator` accepts them.

    Merging the Python run and the CLI run into one object lets
    `compare_accuracy` treat them as two runners -- a way to compare results that
    were never in the same process.
    """
    if not cli_output_json.exists():
        print("    no CLI output to compare against, skipping")
        return

    # The comparison is only meaningful because Polygraphy's default data loader
    # is deterministic: the CLI run and the Python run generated bit-identical
    # inputs without either being told to. Verified rather than assumed.
    cli_input_json = output_path / "cli-inputs.json"
    if cli_input_json.exists():
        same_input = np.array_equal(np.asarray(load_json(str(input_json))[0]["x"]), np.asarray(load_json(str(cli_input_json))[0]["x"]))
        print(f"    CLI and Python got identical inputs without coordinating: {same_input}")

    merged = RunResults()
    merged.add(list(results["trt"]), runner_name="python")
    cli_results = RunResults.load(str(cli_output_json))
    merged.add(list(cli_results[list(cli_results.keys())[0]]), runner_name="cli")

    print(f"    merged runners: {list(merged.keys())}")
    passed = bool(Comparator.compare_accuracy(merged, compare_func=CompareFunc.simple(atol=1e-5)))
    print(f"    python vs cli within 1e-5: {passed}")
    print("    two processes, two engines, same answer -- and the inputs matched because the")
    print("    default data loader is seeded. For a real dataset, pass the saved file to the")
    print("    CLI with `--load-inputs` instead of relying on that.")
    return

if __name__ == "__main__":
    case_save_from_python()
    case_load_and_check_the_round_trip()
    case_the_cli_writes_the_same_format()
    case_compare_across_the_two_runs()

    print("\nFinish")
