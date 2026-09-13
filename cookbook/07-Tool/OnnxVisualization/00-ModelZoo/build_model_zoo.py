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
"""Write every model of the zoo to `model/`, plus a `manifest.json` index.

The models and their expected outliner results live in `model_zoo.py`, which is
also what `../01-BasicUsage/test_outliner.py` imports. This file only puts them on
disk, so nothing downstream depends on the code that first created them.

    python3 build_model_zoo.py            # write model/*.onnx and manifest.json
    python3 build_model_zoo.py --check    # also load every model with onnxruntime
"""

import argparse
import json
import sys

import onnx

from model_zoo import EXPECT, SYNTHETIC, model_path, output_path, torch_model_list

def main() -> int:
    """Build every model, write the manifest, optionally cross check with onnxruntime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Also load every model with onnxruntime")
    args = parser.parse_args()

    model_path.mkdir(exist_ok=True)
    manifest = []

    for builder, description in SYNTHETIC:
        model = builder()
        onnx_file = model_path / f"{builder.__name__}.onnx"
        onnx.save(model, onnx_file)
        manifest.append({"name": builder.__name__, "source": "synthetic", "description": description, "n_node": len(model.graph.node), "size_kb": round(onnx_file.stat().st_size / 1024, 1)})

    for name, builder, description in torch_model_list():
        onnx_file = model_path / f"{name}.onnx"
        model = builder(onnx_file)
        manifest.append({"name": name, "source": "pytorch", "description": description, "n_node": len(model.graph.node), "n_function": len(model.functions), "size_kb": round(onnx_file.stat().st_size / 1024, 1)})

    # The expected outliner answer travels with the model, so the corpus describes
    # itself and `test_outliner.py` cannot drift away from it.
    for entry in manifest:
        expect = EXPECT.get(entry["name"])
        if expect is not None:
            entry["expect"] = {"config": expect["config"], "pattern": expect["pattern"], "note": expect["note"]}

    n_fail = 0
    if args.check:
        import onnxruntime as ort
        for entry in manifest:
            onnx_file = model_path / f"{entry['name']}.onnx"
            try:
                session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
                entry["onnxruntime"] = "pass"
                del session
            except Exception as e:
                entry["onnxruntime"] = f"{type(e).__name__}: {e}"
                n_fail += 1

    missing = [entry["name"] for entry in manifest if "expect" not in entry]
    if missing:  # Not silent: a model with no expected answer is not a test case yet
        print(f"WARNING: no entry in model_zoo.EXPECT for {missing}")
        n_fail += len(missing)

    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"{'name':<26}{'source':<11}{'node':>6}{'KB':>9}  {'expected pattern':<18}description")
    print("-" * 130)
    for entry in manifest:
        pattern = str(entry.get("expect", {}).get("pattern", "?"))
        print(f"{entry['name']:<26}{entry['source']:<11}{entry['n_node']:>6}{entry['size_kb']:>9}  {pattern:<18}{entry['description'][:48]}")
    print("-" * 130)
    print(f"{len(manifest)} models in {model_path}, manifest in {output_path / 'manifest.json'}")
    if args.check:
        print(f"onnxruntime: {len(manifest) - n_fail}/{len(manifest)} load")
    print("The expected patterns above are asserted by ../01-BasicUsage/test_outliner.py, which imports")
    print("the very same builders from model_zoo.py, so a model and its answer cannot drift apart.")
    return 1 if n_fail else 0

if __name__ == "__main__":
    code = main()
    print("\nFinish")
    sys.exit(code)
