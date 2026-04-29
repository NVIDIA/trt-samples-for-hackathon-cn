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
import os
import re
import subprocess
from pathlib import Path
from typing import Union
import logging
import datetime

########################################################################################################################
# Tool functions for Cookbook utilities

def resolve_trt_cookbook_path(start_path: Union[str, Path, None] = None, set_env: bool = True, strict: bool = True) -> Path | None:
    """Resolve cookbook root path from env var or by walking upward from common anchors."""

    def _is_cookbook_root(path: Path) -> bool:
        """Check whether a path looks like the cookbook root directory."""
        return path.is_dir() and (path / "00-Data").is_dir() and (path / "tensorrt_cookbook").is_dir()

    def _iter_parent_paths(start: Path):
        current = start.resolve()
        if current.is_file():
            current = current.parent
        yield current
        for parent in current.parents:
            yield parent

    env_var = "TRT_COOKBOOK_PATH"
    candidate_roots = []
    env_path = os.environ.get(env_var)
    if env_path:
        candidate_roots.append(Path(env_path).expanduser())
    if start_path is not None:
        candidate_roots.append(Path(start_path).expanduser())
    candidate_roots.extend([Path.cwd(), Path(__file__)])
    for candidate in candidate_roots:
        try:
            for maybe_root in _iter_parent_paths(candidate):
                if _is_cookbook_root(maybe_root):
                    if set_env:
                        os.environ[env_var] = str(maybe_root)
                    return maybe_root
        except OSError:
            continue
    if strict:
        raise EnvironmentError("Cannot resolve TRT_COOKBOOK_PATH automatically. "
                               "Please set TRT_COOKBOOK_PATH to the cookbook root (the directory containing 00-Data/).")
    return None

def cookbook_path(*parts, start_path: Union[str, Path, None] = None, b_must_exist: bool = True) -> Path:
    """Build a path under cookbook root with automatic root discovery."""
    root = resolve_trt_cookbook_path(start_path)
    target = root.joinpath(*parts)
    if b_must_exist and not target.exists():
        raise FileNotFoundError(f"Path not found under cookbook root: {target}")
    return target

def grep_used_members(file_path: Path, target_member_set: set, b_print: bool = True) -> set[str]:
    """Scan a source file and return member names referenced via dot access."""
    target_member_set -= {"_pybind11_conduit_v1_"}  # Always skip the pybind member
    text = file_path.read_text(encoding="utf-8")
    used_member = set(re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)", text))

    if b_print:
        covered_member = sorted(target_member_set & used_member)
        uncovered_member = sorted(target_member_set - used_member)
        if len(uncovered_member) > 0:  # No print output if all APIs are covered
            print(f"\n{'=' * 16} Usage coverage in this script")
            print(f"{len(covered_member):2d}/{len(target_member_set):2d} used members: {covered_member}")
            print(f"{len(uncovered_member):2d} uncovered members: {uncovered_member}")
            print()
    return used_member

def print_enumerated_members(enum_class):
    """Print members of an enum-like class with their integer values when possible."""
    print(f"Members of {enum_class.__module__}.{enum_class.__qualname__}:")
    for key, value in enum_class.__members__.items():
        print(f"{int(value):2d} -> {key}")

# For logging

def get_cookbook_logger(name: str | None = None, b_log_stdout: bool = False, log_file: str = "log.log") -> logging.Logger:

    class E8Formatter(logging.Formatter):

        def converter(self, timestamp):
            return datetime.datetime.fromtimestamp(timestamp, datetime.timezone(datetime.timedelta(hours=8)))

        def formatTime(self, record, datefmt=None):
            dt = self.converter(record.created)
            return dt.strftime(datefmt) if datefmt is not None else dt.isoformat()

    formatter = E8Formatter(fmt="[%(asctime)s]%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handlers = []
    if b_log_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        handlers=handlers,
    )
    return logging.getLogger(name)

########################################################################################################################
# Tool functions for package inspection

SKIP_NAMES = {"ctypes", "os", "sys", "tensorrt", "warnings"}
LEAF_TYPES = (int, float, str, list, dict, tuple, set, bool, bytes, bytearray, complex, type(None))

def safe_repr(value, max_len=160):
    """Return a shortened and exception-safe ``repr`` string for ``value``."""
    try:
        text = repr(value)
    except Exception as error:
        text = f"<repr-error: {error}>"
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text

def safe_getattr(obj, name):
    """Get an attribute safely and return ``(ok, value, error_text)``."""
    try:
        return True, getattr(obj, name), None
    except Exception as error:
        return False, None, f"{type(error).__name__}: {error}"

def get_signature_text(target):
    """Return function signature text, or ``(...)`` when unavailable."""
    try:
        return str(inspect.signature(target))
    except Exception:
        return "(...)"

def get_enum_member_int_value(current, target):
    """Extract integer value for enum-like class members when possible."""
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
    """Check whether an object should be recursively expanded for API listing."""
    if inspect.ismodule(target):
        module_name = getattr(target, "__name__", "")
        return module_name.startswith(root_package)

    if inspect.isclass(target):
        module_name = getattr(target, "__module__", "")
        return module_name.startswith(root_package)

    return False

def iter_public_names(obj):
    """Iterate sorted public member names excluding skipped helper names."""
    names = []
    for name in dir(obj):
        if name.startswith("__"):
            continue
        if name in SKIP_NAMES:
            continue
        names.append(name)
    return sorted(names)

def list_api(module_name: str, output_path: Union[str, Path] = ".", max_depth: int = 12):
    """Generate a tree-style API inventory for a module and write it to disk."""
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "unknown")

    output_path = Path(output_path)
    output_file = output_path / f"result-{module_name}-{version}.log"

    lines = [module_name]
    visited = set()

    def walk(current, prefix, depth):
        """Recursively traverse members and append formatted lines."""
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
