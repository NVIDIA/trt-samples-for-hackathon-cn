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
"""Guard `../standalone/`: it must not drift from the package, and it must run.

Two checks:

1. `../standalone/onnx_outliner/` is a copy of `tensorrt_cookbook/onnx_outliner`.
   Only `README.md` and the `_tensorrt_check` body of `outliner.py` are allowed
   to differ, so an upstream change that was never copied over is reported here
   instead of silently shipping stale code.
2. `../standalone/main.py` outlines a model end to end **in a fresh
   interpreter with `tensorrt_cookbook` made unimportable**, which is the only
   honest way to prove the "no install needed" claim.

Returns a non-zero exit code on any failure.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Injected into the child interpreter to make `import tensorrt_cookbook` fail.
BLOCKER = '''
import sys

class _Deny:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name == "tensorrt_cookbook" or name.startswith("tensorrt_cookbook."):
            raise ImportError(f"{name} is blocked by the standalone test")
        return None

sys.meta_path.insert(0, _Deny())
'''

output_path = Path(__file__).resolve().parent
# `standalone/` is a sibling of this directory, not a child: it is meant to be copied out
# on its own, so it does not live inside an example.
standalone_path = output_path.parent / "standalone"
copy_path = standalone_path / "onnx_outliner"
package_path = output_path.parent.parent.parent / "tensorrt_cookbook" / "onnx_outliner"

# `README.md` documents the copy itself, `outliner.py` carries the one intended
# code difference. Everything else has to match byte for byte.
ALLOWED_DIFFERENT = {"README.md", "outliner.py"}

n_fail = 0

def check(b_ok: bool, message: str) -> None:
    """Print one check line and count the failures."""
    global n_fail
    n_fail += not b_ok
    print(f"    [{'PASS' if b_ok else 'FAIL'}] {message}")
    return

def case_no_drift() -> None:
    """Every file of the copy matches the package, apart from the known two."""
    print("case_no_drift")
    if not package_path.exists():  # The copy was taken out of the cookbook, which is the point of it
        print(f"    skipped, {package_path} not found")
        return
    name_copy = {p.name for p in copy_path.iterdir() if p.is_file()}
    name_package = {p.name for p in package_path.iterdir() if p.is_file()}
    check(name_copy == name_package, f"same file list, copy-only={sorted(name_copy - name_package)}, package-only={sorted(name_package - name_copy)}")
    for name in sorted(name_copy & name_package - ALLOWED_DIFFERENT):
        check((copy_path / name).read_bytes() == (package_path / name).read_bytes(), f"{name} identical to the package")
    # `outliner.py` may differ only inside `_tensorrt_check`.
    text_copy = (copy_path / "outliner.py").read_text().split("def _tensorrt_check")
    text_package = (package_path / "outliner.py").read_text().split("def _tensorrt_check")
    check(len(text_copy) == len(text_package) == 2, "outliner.py has exactly one _tensorrt_check in both")
    if len(text_copy) == len(text_package) == 2:
        check(text_copy[0] == text_package[0], "outliner.py identical before _tensorrt_check")
        tail_copy = text_copy[1].split("\ndef outline(")
        tail_package = text_package[1].split("\ndef outline(")
        check(len(tail_copy) == len(tail_package) == 2 and tail_copy[1] == tail_package[1], "outliner.py identical after _tensorrt_check")
    # Prose may still *mention* the package it was copied from, an import may not.
    import_line = [f"{name}:{i}" for name in sorted(p for p in name_copy if p.endswith(".py")) for i, line in enumerate((copy_path / name).read_text().splitlines(), 1) if line.strip().startswith(("import ", "from ")) and ("tensorrt_cookbook" in line or line.strip().startswith(("from ..", "from ...")))]
    check(not import_line, f"the copy imports nothing outside itself, offenders={import_line}")
    # The three hard dependencies must be spelled out, or "copy the directory and
    # `pip install -r requirements.txt`" is not actually enough to run it.
    requirement = (standalone_path / "requirements.txt").read_text() if (standalone_path / "requirements.txt").exists() else ""
    missing = [p for p in ["numpy", "onnx", "onnx_graphsurgeon"] if not any(line.strip() == p for line in requirement.splitlines())]
    check(not missing, f"requirements.txt lists every required package, missing={missing}")
    return

def build_toy_model(onnx_file: Path, n_block: int = 4, size: int = 8) -> None:
    """Write a flat `(MatMul, Add, Relu) x n_block` chain, the smallest thing worth outlining.

    Built with `onnx.helper` rather than taken from `../00-ModelZoo`, because the
    zoo exports from PyTorch and this test is about what a user without the
    cookbook can run.
    """
    import numpy as np
    import onnx

    node, initializer = [], []
    name = "input"
    for i in range(n_block):
        rng = np.random.default_rng(i)
        initializer += [
            onnx.numpy_helper.from_array(rng.random([size, size]).astype(np.float32) - 0.5, f"w{i}"),
            onnx.numpy_helper.from_array(rng.random([size]).astype(np.float32) - 0.5, f"b{i}"),
        ]
        node += [
            onnx.helper.make_node("MatMul", [name, f"w{i}"], [f"m{i}"], f"MatMul{i}"),
            onnx.helper.make_node("Add", [f"m{i}", f"b{i}"], [f"a{i}"], f"Add{i}"),
            onnx.helper.make_node("Relu", [f"a{i}"], [f"r{i}"], f"Relu{i}"),
        ]
        name = f"r{i}"
    graph = onnx.helper.make_graph(
        node,
        "toy",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, size])],
        [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, size])],
        initializer,
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 10  # Some onnxruntime builds still refuse the newest one
    onnx.save(model, onnx_file)
    return

def case_run_without_cookbook() -> None:
    """End to end, in an interpreter where `import tensorrt_cookbook` fails."""
    print("case_run_without_cookbook")
    onnx_file = output_path / "model-toy.onnx"
    build_toy_model(onnx_file)
    output_file = output_path / "model-standalone.onnx"
    with tempfile.TemporaryDirectory() as temp_dir:
        # The developer running this very likely *does* have `tensorrt_cookbook`
        # installed, so hiding it has to be active rather than hoped for: a
        # `sitecustomize.py` on `PYTHONPATH` denies the import outright.
        (Path(temp_dir) / "sitecustomize.py").write_text(BLOCKER)
        # `-P` additionally keeps the cwd off `sys.path`; `sys.path[0]` is then not
        # the script's directory either, which is exactly the case
        # `../standalone/main.py` inserts it by hand for.
        command = [sys.executable, "-P", str(standalone_path / "main.py"), str(onnx_file), "-o", str(output_file), "--max-level", "1"]
        python_path = os.pathsep.join([temp_dir] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
        process = subprocess.run(command, capture_output=True, text=True, cwd="/", env={**os.environ, "PYTHONPATH": python_path})
    print("\n".join(f"        {line}" for line in (process.stdout + process.stderr).splitlines()))
    check(process.returncode == 0, "main.py exits 0")
    check(output_file.exists(), f"{output_file.name} written")
    check("local function(s)" in process.stdout and " 0 local function(s)" not in process.stdout, "at least one function was folded out of the toy model")
    output_file.unlink(missing_ok=True)
    onnx_file.unlink(missing_ok=True)
    return

if __name__ == "__main__":
    case_no_drift()
    case_run_without_cookbook()
    print(f"\n{'Finish' if n_fail == 0 else f'{n_fail} check(s) FAILED'}")
    sys.exit(1 if n_fail else 0)
