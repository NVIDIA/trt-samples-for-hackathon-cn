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
"""Hand the exact inputs Polygraphy used over to `trtexec`.

The two tools do not speak the same language about data:

+ `polygraphy run --save-inputs` writes **one JSON file** holding every input tensor of
  every iteration, names included.
+ `trtexec --loadInputs=name:file` wants **one raw binary file per tensor**, no names, no
  shapes, no dtype -- just bytes in the tensor's own layout.

So "reproduce this Polygraphy failure under trtexec" starts with a format conversion that
is easy to get subtly wrong, and wrong here means trtexec silently runs on garbage. This
script does the conversion and prints the `--loadInputs=` argument to paste.

A from-scratch reimplementation of the idea behind an internal, proprietary
`polygraphy_inputs_to_trtexec.py`; no code was taken from it.

Three things that must be right, and are checked below:

1. **C-contiguous bytes.** A non-contiguous array's `.tobytes()` is still correct, but an
   array that was sliced or transposed on the way in will be written in the *wrong* order
   unless it is made contiguous deliberately. `np.ascontiguousarray` before `.tofile`.
2. **The dtype must survive.** trtexec reads the file into the engine's binding, so the
   file has to be in the engine's dtype. A float64 array from NumPy's default would be
   read as garbage; the file name records the dtype so a human can spot the mismatch.
3. **The byte count must match `volume * itemsize` exactly.** trtexec *does* check this and
   the message is clear, which is the one friendly part of the process:

       Unexpected file size for input file: x.4.1.28.28.float32.raw.
       Note: Input binding size is: 3136 bytes but the file size is 12544 bytes.

   But note what it compares against: the **binding** size, i.e. the shape trtexec is
   running at. Forget `--shapes` and it uses the profile default, so a perfectly good file
   is rejected for a reason that has nothing to do with the file.

And one that costs more time than all three:

4. **The two tools do not share a shape syntax.** Polygraphy writes `x:[4,1,28,28]`,
   trtexec wants `x:4x1x28x28`. Passing Polygraphy's form to trtexec does not produce an
   error message -- trtexec **prints its entire help text and exits non-zero**, which looks
   like "trtexec is broken" rather than "one argument was malformed". This script emits the
   `--shapes=` string in trtexec's syntax alongside `--loadInputs=`, because getting the
   data across without the shape is only half the job.
"""

import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np
from polygraphy.json import load_json

from tensorrt_cookbook import case_mark, cookbook_path

np.random.seed(31193)

output_path = Path(__file__).parent
onnx_file = cookbook_path("00-Data", "model", "model-trained.onnx")
input_json_file = output_path / "data-inputs.json"
raw_directory = output_path / "trtexec_inputs"

INPUT_SHAPE = "x:[4,1,28,28]"

result = OrderedDict()

def polygraphy_inputs_to_raw(json_file: Path, directory: Path) -> OrderedDict:
    """Split a Polygraphy `--save-inputs` JSON into one raw file per tensor.

    Returns `name -> (path, shape, dtype, n_byte)`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    data = load_json(str(json_file))
    iteration = data[0]  # `--save-inputs` stores a list of iterations; take the first

    manifest = OrderedDict()
    for name, value in iteration.items():
        array = np.ascontiguousarray(value)  # See point 1 in the module docstring
        # The file name carries what the format cannot: shape and dtype. trtexec ignores it,
        # a human reading the directory does not.
        safe_name = name.replace("/", "_")
        shape_text = ".".join(str(d) for d in array.shape)
        file_path = directory / f"{safe_name}.{shape_text}.{array.dtype}.raw"
        array.tofile(file_path)
        assert file_path.stat().st_size == array.nbytes, "Short write"
        manifest[name] = (file_path, array.shape, array.dtype, array.nbytes)
    return manifest

@case_mark
def case_save_inputs() -> None:
    """Produce a Polygraphy input JSON to convert."""
    command = ["polygraphy", "run", str(onnx_file), "--onnxrt", "--input-shapes", INPUT_SHAPE, "--save-inputs", str(input_json_file), "--seed", "31193"]
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr[-2000:]
    data = load_json(str(input_json_file))
    print(f"    {input_json_file.name}: {len(data)} iteration(s), tensors {[(k, v.shape, str(v.dtype)) for k, v in data[0].items()]}")
    return

@case_mark
def case_convert() -> None:
    """JSON -> one raw file per tensor, plus the argument to paste."""
    manifest = polygraphy_inputs_to_raw(input_json_file, raw_directory)
    for name, (path, shape, dtype, n_byte) in manifest.items():
        print(f"    {name:<8} -> {path.name:<32} shape={shape} dtype={dtype} bytes={n_byte}")
    argument = ",".join(f"{name}:{path.name}" for name, (path, _, _, _) in manifest.items())
    # trtexec's shape syntax is `name:AxBxC`, NOT Polygraphy's `name:[A,B,C]`
    shape_argument = ",".join(f"{name}:{'x'.join(str(d) for d in shape)}" for name, (_, shape, _, _) in manifest.items())
    print(f"\n    trtexec --loadInputs={argument} \\")
    print(f"            --shapes={shape_argument}")
    print(f"    (Polygraphy writes the shape as {INPUT_SHAPE!r}; trtexec rejects that form by printing its help)")
    result["manifest"] = manifest
    result["argument"] = argument
    result["shape_argument"] = shape_argument
    return

@case_mark
def case_verify_round_trip() -> None:
    """Read the raw files back and prove they are byte-identical to what Polygraphy held.

    This is the check that catches a wrong dtype or a non-contiguous write, both of which
    produce a file of *plausible* size that decodes into different numbers.
    """
    original = load_json(str(input_json_file))[0]
    for name, (path, shape, dtype, n_byte) in result["manifest"].items():
        restored = np.fromfile(path, dtype=dtype).reshape(shape)
        reference = np.ascontiguousarray(original[name])
        assert restored.dtype == reference.dtype, f"{name}: dtype changed"
        assert restored.shape == reference.shape, f"{name}: shape changed"
        assert np.array_equal(restored, reference), f"{name}: bytes differ"
        expected_byte = int(np.prod(shape)) * np.dtype(dtype).itemsize
        assert n_byte == expected_byte, f"{name}: {n_byte} B on disk, engine will read {expected_byte} B"
        print(f"    {name:<8} round trip exact, {n_byte} B = volume {int(np.prod(shape))} x itemsize {np.dtype(dtype).itemsize}")
    return

@case_mark
def case_run_trtexec() -> None:
    """Feed the converted files to trtexec for real, so the format claim is tested."""
    command = ["trtexec", f"--onnx={onnx_file}", f"--loadInputs={result['argument']}", f"--shapes={result['shape_argument']}", "--iterations=1", "--warmUp=0"]
    process = subprocess.run(command, capture_output=True, text=True, cwd=str(raw_directory))
    passed = "PASSED" in process.stdout
    print(f"    trtexec exit={process.returncode}, PASSED in output={passed}")
    for line in process.stdout.splitlines():
        if "Using values loaded" in line:
            print(f"    {line.split('] ')[-1]}")
    if not passed:
        print("\n".join(process.stdout.splitlines()[-15:]))
    assert process.returncode == 0 and passed, "trtexec did not accept the converted inputs"

    # And demonstrate the failure mode, so the behaviour is in the log rather than only in a comment
    bad_command = ["trtexec", f"--onnx={onnx_file}", f"--loadInputs={result['argument']}", f"--shapes={INPUT_SHAPE}", "--iterations=1", "--warmUp=0"]
    bad_process = subprocess.run(bad_command, capture_output=True, text=True, cwd=str(raw_directory))
    printed_help = "=== Help ===" in bad_process.stdout
    print(f"    same run with Polygraphy's shape syntax: exit={bad_process.returncode}, trtexec printed its whole help={printed_help}")
    assert printed_help, "Expected trtexec to reject the bracket shape syntax by printing help"
    return

def main() -> None:
    case_save_inputs()
    case_convert()
    case_verify_round_trip()
    case_run_trtexec()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
