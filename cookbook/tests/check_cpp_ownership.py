#!/usr/bin/env python3
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
"""Check that every C++ example deletes the TensorRT objects it owns.

The cookbook deliberately does **not** use RAII wrappers in its C++ examples (that was tried and
reverted): creations are paired with explicit `delete`s so a reader can see the lifetime. This
script is what keeps that pairing honest, because a missing `delete` is invisible in a passing run.

Two things it knows that a naive grep does not:

+ `IOptimizationProfile` is **not** owned by the caller. `NvInfer.h` says the builder "retains
  ownership of the created optimization profile ... the users must not attempt to delete the
  returned pointer", so deleting it would be the bug.
+ `buildSerializedNetwork` returns an `IHostMemory` the caller owns, even though it does not look
  like a factory call.

`compute-sanitizer --tool memcheck --leak-check full` is the dynamic complement; it catches leaked
*device* memory but only along the path actually executed.

```bash
python3 tests/check_cpp_ownership.py
```
"""

import re
import sys
from pathlib import Path

# TensorRT factory calls whose result the caller owns and must `delete`.
OWNED_FACTORY = {
    "createInferBuilder": "IBuilder",
    "createNetworkV2": "INetworkDefinition",
    "createBuilderConfig": "IBuilderConfig",
    "buildSerializedNetwork": "IHostMemory",
    "createInferRuntime": "IRuntime",
    "deserializeCudaEngine": "ICudaEngine",
    "createExecutionContext": "IExecutionContext",
    "createParser": "IParser",
    "createRefitter": "IRefitter",
    "createEngineInspector": "IEngineInspector",
}
# Explicitly NOT owned by the caller - deleting these is the bug.
NOT_OWNED_FACTORY = {"createOptimizationProfile": "IOptimizationProfile"}

SKIP_DIR = ("91-OldStuff", "92-LocalFile", ".venv", "build", "dist")

def check_one(path: Path) -> list[str]:
    """Return a list of complaints about one C++ file (empty means clean)."""
    source = path.read_text()
    created: dict[str, tuple[str, int]] = {}
    for line_number, line in enumerate(source.splitlines(), 1):
        if line.lstrip().startswith("//"):
            continue
        for call, type_name in OWNED_FACTORY.items():
            if call + "(" not in line:
                continue
            match = re.search(r"(\w+)\s*=\s*[^=]*" + call, line)
            if match:
                created.setdefault(match.group(1), (type_name, line_number))

    deleted = set(re.findall(r"\bdelete\s+(\w+)\s*;", source))

    complaint_list = []
    for variable, (type_name, line_number) in created.items():
        if variable not in deleted:
            complaint_list.append(f"{path}:{line_number}: `{variable}` ({type_name}) is never deleted")
    for call, type_name in NOT_OWNED_FACTORY.items():
        for match in re.finditer(r"(\w+)\s*=\s*[^=]*" + call, source):
            if match.group(1) in deleted:
                complaint_list.append(f"{path}: `{match.group(1)}` ({type_name}) is deleted, but the builder owns it")
    return complaint_list

def main() -> int:
    base_dir = Path(__file__).resolve().parent.parent
    file_list = sorted(p for p in base_dir.rglob("*.cpp") if not any(s in str(p) for s in SKIP_DIR))

    complaint_list = []
    for path in file_list:
        complaint_list += check_one(path)

    for complaint in complaint_list:
        print(f"[E] {complaint}")
    print(f"Checked {len(file_list)} C++ files, {len(complaint_list)} problem(s)")
    return 1 if complaint_list else 0

if __name__ == "__main__":
    sys.exit(main())
