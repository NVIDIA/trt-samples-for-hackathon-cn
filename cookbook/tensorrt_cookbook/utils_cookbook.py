# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import importlib
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Union
import logging
import datetime
import tensorrt as trt
from typing import Callable, Union, ParamSpec

########################################################################################################################
# Tool functions for Cookbook utilities

def resolve_trt_cookbook_path(start_path: Union[str, Path, None] = None, set_env: bool = True, strict: bool = True) -> Path | None:
    """Resolve cookbook root path from environment variable or by walking upward from common anchors."""

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

# TRT wrapper/callback members that are excluded from the inspected object's public member lists.
# _pybind11_conduit_v1_ is intentionally NOT here: it is always excluded via the inline | {"_pybind11_conduit_v1_"}
# below, keeping it separate makes the intent explicit.
_COMMON_MEMBER_EXCLUDE_SET = {
    "logger",
    "builder",
    "algorithm_selector",
    "error_recorder",
    "gpu_allocator",
    "int8_calibrator",
    "progress_monitor",
    "_pybind11_conduit_v1_",
}

# For Layer API examples
_LAYER_MEMBER_EXCLUDE_SET = {
    "get_input",
    "get_output_type",
    "get_output",
    "metadata",
    "name",
    "num_inputs",
    "num_outputs",
    "num_ranks",
    "output_type_is_set",
    "precision_is_set",
    "precision",
    "reset_output_type",
    "reset_precision",
    "set_input",
    "set_output_type",
    "type",
}

def analyze_public_members(obj_instance: object = None, obj_class: object = None, exclude_set: set[str] | None = None, b_print: bool = True) -> set[str]:
    """Analyze public members of a TRT object/class and optionally print a summary."""
    _exclude = (exclude_set or set()) | {"_pybind11_conduit_v1_"}

    def _collect(target: object) -> tuple[list[str], list[str], list[str], set[str]]:
        pub = set(x for x in dir(target) if not x.startswith("__"))
        cb = sorted(pub & _COMMON_MEMBER_EXCLUDE_SET - _exclude)
        ca = sorted(x for x in pub - _COMMON_MEMBER_EXCLUDE_SET - _exclude if callable(getattr(target, x)))
        at = sorted(pub - set(cb) - set(ca) - _exclude)
        return cb, ca, at, set(cb + ca + at)

    def _print(target: object, cb: list[str], ca: list[str], at: list[str]) -> None:
        print(f"\n{'='* 16} Public members of {target} ({'class' if isinstance(target, type) else 'instance'})")
        print(f"{len(cb):2d} Callback members: {cb}")
        print(f"{len(ca):2d} Callable methods: {ca}")
        print(f"{len(at):2d} Non-callable attributions: {at}")

    if obj_instance is None:
        assert obj_class is not None, "Either obj_instance or obj_class must be provided"
        cb, ca, at, public = _collect(obj_class)
        if b_print:
            _print(obj_class, cb, ca, at)
        return public

    if isinstance(obj_instance, trt.ILayer):
        _exclude |= _LAYER_MEMBER_EXCLUDE_SET

    assert (obj_class is None) or (obj_class is obj_instance.__class__), f"{obj_class} must be the class of {obj_instance}"
    cb, ca, at, instance_public = _collect(obj_instance)
    _, _, _, class_public = _collect(obj_instance.__class__)

    if b_print:
        _print(obj_instance, cb, ca, at)
        class_only = sorted(class_public - instance_public)
        instance_only = sorted(instance_public - class_public)
        if class_only or instance_only:
            print(f"\n{'=' * 16} Class/Object difference")
            print(f"{len(class_only):2d} class-only members: {class_only}")
            print(f"{len(instance_only):2d} instance-only members: {instance_only}")

    return instance_public

def grep_used_members(file_path: Path, member_set: set, b_print: bool = True) -> set[str]:
    """Scan a source file and return member names referenced via dot access."""
    local_member_set = member_set - {"_pybind11_conduit_v1_"}  # Always skip the pybind member
    text = file_path.read_text(encoding="utf-8")
    used_member = set(re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)", text))

    if b_print:
        covered_member = sorted(local_member_set & used_member)
        uncovered_member = sorted(local_member_set - used_member)
        if len(uncovered_member) > 0:  # No print output if all APIs are covered
            print(f"\n{'=' * 16} Usage coverage in this script")
            print(f"{len(covered_member):2d}/{len(local_member_set):2d} used members: {covered_member}")
            print(f"{len(uncovered_member):2d} uncovered members: {uncovered_member}")
            print()
    return used_member

def check_api_coverage(
    obj_instance: object = None,
    obj_class: object = None,
    exclude_set: set[str] | None = None,
    b_print: bool = True,
) -> set[str]:
    """Analyze public members of an object/class and check coverage in the caller's source file."""
    caller_file = Path(inspect.stack()[1].filename)
    public_member = analyze_public_members(obj_instance, obj_class, exclude_set, b_print)
    return grep_used_members(caller_file, public_member, b_print)

def print_enumerated_members(enum_class):
    """Print members of an enum-like class with their integer values when possible."""
    print(f"Members of {enum_class.__module__}.{enum_class.__qualname__}:")
    for key, value in enum_class.__members__.items():
        print(f"{int(value):2d} -> {key}")

# For logging

def get_cookbook_logger(name: str | None = None, b_log_stdout: bool = False, log_file: str = "log.log") -> logging.Logger:
    """Create a UTC+8 logger and attach file/stdout handlers."""

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
        handlers=handlers,
    )
    return logging.getLogger(name)

P = ParamSpec("P")

def run_once(f: Callable[P, None]) -> Callable[P, None]:

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        if not wrapper.has_run:  # type: ignore[attr-defined]
            wrapper.has_run = True  # type: ignore[attr-defined]
            return f(*args, **kwargs)

    wrapper.has_run = False  # type: ignore[attr-defined]
    return wrapper

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

    if module_name in ["tensorrt"]:  # Normla workflow
        module_for_stubgen = f"{module_name}"
    if module_name in ["polygraphy"]:  # Some module needs repeat its name as submodule
        module_for_stubgen = f"{module_name}.{module_name}"
    else:
        module_for_stubgen = f"{module_name}"
        print(f"Module {module_name} is not tested by this script, use default configuration.")

    subprocess.run(
        [sys.executable, "-m", "pybind11_stubgen", "--ignore-all-errors", module_for_stubgen, "-o", str(stub_output_dir)],
        check=False,
    )

    print(f"Text output to {output_file}")
    print(f"Stub output to {stub_output_dir}/")
    return lines

def build_copyright(directory: Path, depth: int = 100):

    ################################ Tool members
    type_p = (".py", )
    type_c = (".c", ".cpp", ".h", ".hpp", ".cu", ".cuh")
    type_sh = (".sh", )

    pattern_p_start = r"\A(?:(?:#(?!\!).*\n)|(?:#\n)|(?:# \n))+\n*"

    pattern_c_start = r"\A/\*[\s\S]*?\*/\n*"

    text = """# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
"""

    text_p = text + "\n"

    text_c = "/*\n" + text.replace("#", " *") + " */\n\n"

    dry_run = False

    exclude_list = [".git", ".pytest_cache", ".vscode", "dist", "tensorrt_cookbook.egg-info"]

    def update(f):

        if f.name.endswith(type_p):
            pattern = re.compile(pattern_p_start)
            header = text_p
            shebang = re.compile(r'^(#![^\n]*\n)')
            default_shebang = ""
        elif f.name.endswith(type_c):
            pattern = re.compile(pattern_c_start)
            header = text_c
            shebang = None
            default_shebang = ""
        elif f.name.endswith(type_sh):
            pattern = re.compile(pattern_p_start)
            header = text_p
            shebang = re.compile(r'^(#![^\n]*\n)')
            default_shebang = "#!/bin/bash\n\n"
        else:
            print(f"Skip: {f}")
            return

        with open(f, "r", encoding="utf-8") as file:
            data = file.read()

            prefix = ""
            body = data
            if shebang:
                shebang_match = shebang.match(data)
                if shebang_match:
                    prefix = shebang_match.group(1)
                    body = data[shebang_match.end():]
                else:
                    prefix = default_shebang

            match = pattern.search(body)
            if match:
                print(f"Fix : {f}")
                new_body = pattern.sub(header, body, count=1)
            else:
                print(f"Add : {f}")
                new_body = header + body

            new_data = prefix + new_body

            if not dry_run and new_data != data:
                with open(f, "w", encoding="utf-8") as file:
                    file.write(new_data)

    ################################################################

    for f in sorted(directory.glob("*")):
        if f.name in exclude_list:
            continue
        if f.is_dir() and (depth > 0):
            build_copyright(f, depth - 1)
        elif f.name.endswith(type_p + type_c + type_sh):
            update(f)

def build_readme(path: Path):
    max_lines_from_child_readme_file: int = 3

    print(f"Build README.md for {path.name}")
    output = f"# {path.name}\n"

    readme_outline_file = path / "README.outline.txt"
    if readme_outline_file.exists():
        with open(readme_outline_file, "r") as file:
            outline = file.read()
            output += "\n" + outline

    for sub_dir in sorted(path.glob("*/")):
        if not sub_dir.is_dir() or sub_dir.name.startswith(".") or sub_dir.name in ["__pycache__", "dist", "include", "tensorrt_cookbook.egg-info"]:
            continue
        readme_outline_file = sub_dir / "README.outline.txt"
        readme_file = sub_dir / "README.md"
        if readme_outline_file.exists():
            with open(readme_outline_file, "r") as file:
                outline = file.read()
                output += f"\n## {sub_dir.name}\n\n{outline}"
            build_readme(sub_dir)
        elif readme_file.exists():
            with open(readme_file, "r") as file:
                lines = file.readlines()
                output += f"\n#{''.join(lines[:max_lines_from_child_readme_file])}"
            # Do not walk deeper if no outline file
        else:
            print(f"Skip {sub_dir.name} since no README found")

    with open(path / "README.md", "w") as f:
        f.write(output)
