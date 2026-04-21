# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import importlib
import inspect
import re
import subprocess
from pathlib import Path
from typing import Union

########################################################################################################################
# Tool functions for Cookbook utilities

def grep_used_members(file_path: Path, target_member_set: set, b_print: bool = True) -> set[str]:
    text = file_path.read_text(encoding="utf-8")
    used_member = set(re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)", text))

    if b_print:
        covered_member = sorted(target_member_set & used_member)
        uncovered_member = sorted(target_member_set - used_member)
        if len(uncovered_member) > 0:  # No print output if all APIs are covered
            print(f"\n{'=' * 64} Usage coverage in this script")
            print(f"{len(covered_member):2d}/{len(target_member_set):2d} used members: {covered_member}")
            print(f"{len(uncovered_member):2d} uncovered members: {uncovered_member}")

    return used_member

########################################################################################################################
# Tool functions for package inspection

SKIP_NAMES = {"ctypes", "os", "sys", "tensorrt", "warnings"}
LEAF_TYPES = (int, float, str, list, dict, tuple, set, bool, bytes, bytearray, complex, type(None))

def safe_repr(value, max_len=160):
    try:
        text = repr(value)
    except Exception as error:
        text = f"<repr-error: {error}>"
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text

def safe_getattr(obj, name):
    try:
        return True, getattr(obj, name), None
    except Exception as error:
        return False, None, f"{type(error).__name__}: {error}"

def get_signature_text(target):
    try:
        return str(inspect.signature(target))
    except Exception:
        return "(...)"

def get_enum_member_int_value(current, target):
    if not inspect.isclass(current):
        return None
    if inspect.isclass(target) or inspect.ismodule(target) or inspect.isroutine(target):
        return None

    try:
        if not isinstance(target, current):
            return None
    except Exception:
        return None

    try:
        return int(target)
    except Exception:
        return None

def is_expandable(target, root_package):
    if inspect.ismodule(target):
        module_name = getattr(target, "__name__", "")
        return module_name.startswith(root_package)

    if inspect.isclass(target):
        module_name = getattr(target, "__module__", "")
        return module_name.startswith(root_package)

    return False

def iter_public_names(obj):
    names = []
    for name in dir(obj):
        if name.startswith("__"):
            continue
        if name in SKIP_NAMES:
            continue
        names.append(name)
    return sorted(names)

def list_api(module_name: str, output_path: Union[str, Path] = ".", max_depth: int = 12):
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "unknown")

    output_path = Path(output_path)
    output_file = output_path / f"result-{module_name}-{version}.log"

    lines = [module_name]
    visited = set()

    def walk(current, prefix, depth):
        names = iter_public_names(current)
        for index, name in enumerate(names):
            is_last = index == len(names) - 1
            branch = "└─" if is_last else "├─"
            line_prefix = f"{prefix}{branch} {name}"

            ok, target, error = safe_getattr(current, name)
            if not ok:
                lines.append(f"{line_prefix}: <unavailable: {error}>")
                continue

            if isinstance(target, LEAF_TYPES):
                lines.append(f"{line_prefix}: {safe_repr(target)}")
                continue

            enum_member_int_value = get_enum_member_int_value(current, target)
            if enum_member_int_value is not None:
                lines.append(f"{line_prefix}: <{type(target).__name__}>, value: {enum_member_int_value}")
                continue

            if type(target).__name__ == "pybind11_static_property":
                try:
                    value = target.fget(target)
                    lines.append(f"{line_prefix}: {safe_repr(value)}")
                except Exception as prop_error:
                    lines.append(f"{line_prefix}: <property-error: {type(prop_error).__name__}: {prop_error}>")
                continue

            if inspect.isroutine(target):
                lines.append(f"{line_prefix}{get_signature_text(target)}")
                continue

            if isinstance(target, property):
                lines.append(f"{line_prefix}: <property>")
                continue

            if depth >= max_depth:
                lines.append(f"{line_prefix}: <max-depth-reached>")
                continue

            if is_expandable(target, module_name):
                object_id = id(target)
                if object_id in visited:
                    lines.append(f"{line_prefix}: <visited>")
                    continue

                lines.append(line_prefix)
                visited.add(object_id)
                child_prefix = prefix + ("   " if is_last else "│  ")
                walk(target, child_prefix, depth + 1)
            else:
                lines.append(f"{line_prefix}: <{type(target).__name__}>")

    visited.add(id(module))
    walk(module, "", 0)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stub_output_dir = output_file.with_suffix("")
    module_for_stubgen = f"{module_name}.{module_name}"
    subprocess.run(
        ["pybind11-stubgen", "--ignore-all-errors", module_for_stubgen, "-o", str(stub_output_dir)],
        check=False,
    )

    return lines
