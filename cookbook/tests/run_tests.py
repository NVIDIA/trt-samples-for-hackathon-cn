#!/usr/bin/env python3
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
"""
Unified example runner for cookbook.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".venv",
    "09-TensorRT-LLM",
    "91-OldStuff",
    "99-Todo",
    "dist",
    "include",
    "tensorrt_cookbook.egg-info",
    "tensorrt_cookbook",
    "tests",
}
ROOT = Path(__file__).resolve().parents[1]

#: Tags that mark an example as needing a deep-learning framework beyond pyTorch. These are
#: skipped by default and run only when the matching `--framework` is passed, so the common case
#: -- a container with torch and nothing else -- does not have to see them fail. `--framework X`
#: owns these cases in both directions: it selects them, and it overrides `enabled: false` for
#: them, because that is the flag their YAML currently uses to keep them out of a default sweep.
FRAMEWORK_TAGS = {
    "tensorflow",
    "tensorflow1",
    "jax",
    "paddlepaddle",
    "paddle",
}

#: A change under one of these makes every example suspect, so `--since` selects everything.
#: `00-Data/` is here because it produces the models and datasets the other sections consume.
CHANGE_SHARED_PREFIXES = (
    "00-Data/",
    "include/",
    "tensorrt_cookbook/",
    "tests/",
)

#: Same, for exact files at the cookbook root.
CHANGE_SHARED_FILES = {
    ".pre-commit-config.yaml",
    "build-Copyright-and-README.py",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup_environment.sh",
}

#: A change to one of these cannot alter what an example does, so it selects nothing by itself.
CHANGE_DOC_SUFFIXES = (".md", ".txt")

os.environ.setdefault("TRT_COOKBOOK_PATH", str(ROOT))
root_env = dict(os.environ)

@dataclass
class ExampleSpec:
    path: Path
    relpath: str
    name: str
    enabled: bool = True
    tags: list[str] = field(default_factory=list)
    timeout: int | None = None
    env: dict[str, str] = field(default_factory=dict)
    pre: list[str] = field(default_factory=list)
    run: list[str] = field(default_factory=list)
    post: list[str] = field(default_factory=list)
    clean: list[str] = field(default_factory=list)

@dataclass
class CaseResult:
    relpath: str
    status: str
    elapsed_s: float
    command: str = ""
    returncode: int = 0
    error: str = ""

def _load_yaml(yaml_file: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Found {yaml_file}, but PyYAML is not installed. "
                           "Please run `pip install pyyaml` (or add it to requirements).") from e

    data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{yaml_file}: root must be a mapping")
    return data

def _to_list(value: Any, field_name: str, yaml_file: Path) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    raise ValueError(f"{yaml_file}: field `{field_name}` must be string or list[string]")

def _to_env(value: Any, yaml_file: Path) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{yaml_file}: field `env` must be mapping[string,string]")
    env: dict[str, str] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            raise ValueError(f"{yaml_file}: env key must be string")
        if not isinstance(v, (str, int, float, bool)):
            raise ValueError(f"{yaml_file}: env value must be scalar for key `{k}`")
        env[k] = str(v)
    return env

# ------------------------------------------------------------------------------------------------
# Change-based selection (`--since`)
#
# Adapted from the idea in `tools/test_impact.py` of NVIDIA/TensorRT-Model-Connect (Apache-2.0):
# classify every changed path as example-owned, shared or documentation, and derive the test set
# from that instead of running everything. Their unit of ownership is a model family; ours is an
# example directory, which makes the mapping simpler -- but the two risky parts are the same, and
# both are handled below: an unrecognised path must widen the selection rather than narrow it, and
# the changed-file list is external input that has to be validated before it is used as a path.
# ------------------------------------------------------------------------------------------------

def _safe_relative(path: str) -> str:
    """Normalise a changed path, rejecting anything that could escape the cookbook.

    The list comes from `git`, so it is not attacker-controlled in normal use -- but it is still
    external input that gets turned into filesystem paths, and an absolute path or a `..` would
    silently select the wrong thing.
    """
    normalized = PurePosixPath(path.strip().replace("\\", "/")).as_posix().removeprefix("./")
    candidate = PurePosixPath(normalized)
    # `PurePosixPath("").as_posix()` is ".", so the empty string has to be rejected by name --
    # a truthiness check alone lets it through.
    if normalized in ("", ".") or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe changed path: {path!r}")
    return normalized

def _git_changed_files(ref: str, base_dir: Path) -> list[str]:
    """Files that differ from `ref`, plus untracked ones, relative to the cookbook directory.

    The working tree is included on purpose: the question being answered is "what should I run
    before committing", not "what did this commit touch".
    """

    def run(*command: str) -> list[str]:
        completed = subprocess.run(["git", *command], cwd=str(base_dir), capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"git {' '.join(command)} failed: {completed.stderr.strip()[:200]}")
        return [line for line in completed.stdout.splitlines() if line.strip()]

    repo_root = Path(run("rev-parse", "--show-toplevel")[0])
    changed = set(run("diff", "--name-only", ref))
    changed |= set(run("ls-files", "--others", "--exclude-standard"))

    # git reports paths from the repository root; the cookbook may be a sub-directory of it.
    result = []
    for entry in changed:
        absolute = repo_root / entry
        try:
            result.append(_safe_relative(absolute.relative_to(base_dir).as_posix()))
        except ValueError:
            continue  # Outside the cookbook; nothing here can be selected by it
    return sorted(set(result))

def _classify_changes(changed_file_list: list[str], relpath_list: list[str]) -> tuple[str, set[str], list[str]]:
    """Map changed files to the examples that must run.

    Returns `(scope, selected_relpaths, notes)` where scope is "all", "selected" or "none".
    """
    # Longest first, so `02-API/Layer/Cast` wins over a hypothetical `02-API/Layer`.
    known = sorted(relpath_list, key=len, reverse=True)
    selected: set[str] = set()
    notes: list[str] = []

    for entry in changed_file_list:
        if entry.startswith(CHANGE_SHARED_PREFIXES) or entry in CHANGE_SHARED_FILES:
            notes.append(f"{entry}: shared -> everything")
            return "all", set(relpath_list), notes

        owner = next((rel for rel in known if entry == rel or entry.startswith(rel + "/")), None)
        if owner is not None:
            if not entry.endswith(CHANGE_DOC_SUFFIXES):
                selected.add(owner)
                notes.append(f"{entry}: owned by {owner}")
            else:
                notes.append(f"{entry}: documentation inside {owner}, selects nothing")
            continue

        if entry.endswith(CHANGE_DOC_SUFFIXES):
            notes.append(f"{entry}: documentation, selects nothing")
            continue

        # Not documentation, not shared, and no example owns it. Widening is the safe direction:
        # a path we have not taught this function about is exactly when we should not be clever.
        notes.append(f"{entry}: unrecognised -> everything (widening on purpose)")
        return "all", set(relpath_list), notes

    if not selected:
        return "none", set(), notes
    return "selected", selected, notes

def _normalize_rel(path: Path, base_dir: Path) -> str:
    return path.relative_to(base_dir).as_posix()

def _path_match(relpath: str, patterns: list[str] | None) -> bool:
    """Does `relpath` match any of `patterns`? No patterns means "everything".

    `fnmatch` rather than `PurePath.match`, because the latter is wrong here in two ways for the
    patterns people actually type. It anchors on the *right*, so `Cast` alone would match
    `02-API/Layer/Cast`; and before Python 3.13 its `**` is just `*` and never crosses a `/`, so
    `--include "02-API/**"` matched nothing at all under `02-API/Layer/`. With `fnmatch` the `*`
    does cross separators, which is what `02-API/**` is obviously meant to mean.
    """
    if not patterns:
        return True
    return any(fnmatch.fnmatchcase(relpath, pattern) for pattern in patterns)

def _discover_examples(base_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base_dir):
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_SKIP_DIRS and not d.startswith(".")]

        current = Path(dirpath)

        has_main = "main.py" in filenames
        has_yaml = "unit_test.yaml" in filenames
        has_skip = ".skip_unit_test" in filenames
        if (has_main or has_yaml) and not has_skip:  # Basic condition to enable unit test
            candidates.append(current)

    candidates.sort()
    return candidates

def _build_spec(example_dir: Path, base_dir: Path, default_timeout: int | None) -> ExampleSpec | None:
    relpath = _normalize_rel(example_dir, base_dir)
    yaml_file = example_dir / "unit_test.yaml"

    if yaml_file.exists():
        cfg = _load_yaml(yaml_file)

        version = cfg.get("version", 1)
        if version != 1:
            raise ValueError(f"{yaml_file}: unsupported version {version}, expected 1")

        spec = ExampleSpec(
            path=example_dir,
            relpath=relpath,
            name=str(cfg.get("name", relpath)),
            enabled=bool(cfg.get("enabled", True)),
            tags=_to_list(cfg.get("tags", []), "tags", yaml_file),
            timeout=cfg.get("timeout", default_timeout),
            env=_to_env(cfg.get("env"), yaml_file),
            pre=_to_list(cfg.get("pre"), "pre", yaml_file),
            run=_to_list(cfg.get("run"), "run", yaml_file),
            post=_to_list(cfg.get("post"), "post", yaml_file),
            clean=_to_list(cfg.get("clean"), "clean", yaml_file),
        )

        if spec.timeout is not None:
            spec.timeout = int(spec.timeout)

        if spec.enabled and not spec.run:
            raise ValueError(f"{yaml_file}: field `run` cannot be empty")
        return spec

    if (example_dir / "main.py").exists():
        return ExampleSpec(
            path=example_dir,
            relpath=relpath,
            name=relpath,
            timeout=default_timeout,
            # `2>&1` is not optional: without it a traceback goes to the terminal and never reaches
            # the log file, so a failure found in a sweep cannot be diagnosed afterwards.
            run=["python3 main.py > log-main.py.log 2>&1"],
        )

    return None

def _run_command(cmd: str, cwd: Path, env: dict[str, str], timeout: int | None, dry_run: bool) -> tuple[int, str]:
    if dry_run:
        return 0, ""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            shell=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, ""
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"

def _run_case(spec: ExampleSpec, args: argparse.Namespace) -> CaseResult:
    start = time.perf_counter()
    merged_env = dict(root_env)
    merged_env.update(spec.env)

    timeout = spec.timeout if spec.timeout is not None else args.timeout

    steps = []
    steps.extend(spec.pre)
    steps.extend(spec.run)
    steps.extend(spec.post)

    print(f"\n=== [{spec.relpath}] ===")
    for cmd in steps:
        print(f"$ {cmd}")
        code, err = _run_command(cmd, spec.path, merged_env, timeout, args.dry_run)
        if code != 0:
            elapsed = time.perf_counter() - start
            return CaseResult(
                relpath=spec.relpath,
                status="failed",
                elapsed_s=elapsed,
                command=cmd,
                returncode=code,
                error=err,
            )

    if args.clean and spec.clean:
        for cmd in spec.clean:
            print(f"$ {cmd}")
            code, err = _run_command(cmd, spec.path, merged_env, timeout, args.dry_run)
            if code != 0:
                print(f"[W] clean failed: {cmd}", file=sys.stderr)

    elapsed = time.perf_counter() - start
    return CaseResult(relpath=spec.relpath, status="passed", elapsed_s=elapsed)

def _filter_specs(specs: list[ExampleSpec], args: argparse.Namespace, change_selection: set[str] | None = None) -> list[ExampleSpec]:
    wanted_framework = set(args.framework)
    run_every_framework = "all" in wanted_framework

    selected: list[ExampleSpec] = []
    for spec in specs:
        framework_tags = set(spec.tags) & FRAMEWORK_TAGS

        # `--framework` is exclusive, not additive: passing it runs the framework-gated cases and
        # nothing else. That is the workflow it exists for -- finish a default sweep, install
        # TensorFlow by hand, then run only what that unlocked, without repeating 257 examples.
        # To get both, run the command twice.
        if wanted_framework:
            if not (framework_tags and (run_every_framework or framework_tags & wanted_framework)):
                continue
            # The `enabled` check is deliberately skipped for these. The framework examples carry
            # `enabled: false` precisely to stay out of a default sweep, which is the job this
            # flag now does properly, so honouring it here would make the flag select nothing.
        elif framework_tags or not spec.enabled:
            continue

        if change_selection is not None and spec.relpath not in change_selection:
            continue
        if args.case and spec.relpath not in args.case:
            continue
        if not _path_match(spec.relpath, args.include):
            continue
        if args.exclude and _path_match(spec.relpath, args.exclude):
            continue
        if args.tags and not set(spec.tags).intersection(args.tags):
            continue
        if args.exclude_tags and set(spec.tags).intersection(args.exclude_tags):
            continue

        selected.append(spec)
    return selected

def main() -> int:
    parser = argparse.ArgumentParser(description="Run cookbook examples via unified runner")

    parser.add_argument("--case", action="append", help="run exact relative case path (repeatable)")
    # `default=None`, not `default=["**"]`: `action="append"` appends to the default rather than
    # replacing it, so a `["**"]` default turned every `--include` into `["**", <yours>]` -- and
    # `**` matches everything, so the flag silently did nothing and the whole suite ran.
    parser.add_argument("--include", action="append", default=None, help="glob include pattern on relative path, e.g. '02-API/**' (repeatable)")
    parser.add_argument("--exclude", action="append", default=[], help="glob exclude pattern on relative path")
    parser.add_argument("--tags", action="append", default=[], help="only run cases with any of these tags")
    parser.add_argument("--exclude-tags", action="append", default=[], help="skip cases with these tags")
    parser.add_argument("--timeout", type=int, default=1800, help="default timeout (seconds) per command")
    parser.add_argument("--fail-fast", action="store_true", help="stop at first failed case")
    parser.add_argument("--clean", action="store_true", help="run clean commands")

    parser.add_argument("--list", action="store_true", help="list discovered cases and exit")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument("--summary-json", type=Path, help="write summary report as JSON")

    parser.add_argument(
        "--since",
        metavar="REF",
        help="only run examples affected by changes against REF (a commit, tag or branch). "
        "Includes uncommitted and untracked files. A change under tensorrt_cookbook/, include/, "
        "tests/, 00-Data/ or a shared root file selects everything; a path no example owns also "
        "selects everything, on purpose.",
    )
    parser.add_argument("--explain-since", action="store_true", help="with --since, print why each changed file selected what it did")
    parser.add_argument(
        "--framework",
        action="append",
        default=[],
        metavar="NAME",
        help="run examples needing this framework, which are skipped by default: " + ", ".join(sorted(FRAMEWORK_TAGS)) + ", or 'all' (repeatable). These are the examples that need a package beyond pyTorch, so "
        "install it first, then pass this flag to run them incrementally.",
    )

    args = parser.parse_args()
    if not ROOT.exists():
        print(f"[ERROR] root does not exist: {ROOT}", file=sys.stderr)
        return 2

    candidates = _discover_examples(ROOT)

    specs: list[ExampleSpec] = []
    for c in candidates:
        try:
            spec = _build_spec(c, ROOT, args.timeout)
            if spec is not None:
                specs.append(spec)
        except Exception as e:
            print(f"[ERROR] {c}: {e}", file=sys.stderr)
            return 2

    change_selection: set[str] | None = None
    if args.since:
        try:
            changed_file_list = _git_changed_files(args.since, ROOT)
        except (RuntimeError, ValueError) as e:
            print(f"[ERROR] --since {args.since}: {e}", file=sys.stderr)
            return 2
        scope, change_selection, notes = _classify_changes(changed_file_list, [s.relpath for s in specs])
        print(f"[since {args.since}] {len(changed_file_list)} changed file(s) -> scope={scope}, {len(change_selection)} example(s)")
        if args.explain_since:
            for note in notes:
                print(f"  {note}")
        if scope == "all":
            change_selection = None  # No narrowing; fall through to the ordinary filters

    specs = _filter_specs(specs, args, change_selection)

    if args.list:
        print("Discovered examples:")
        for spec in specs:
            print(f"- {spec.relpath} | tags={','.join(spec.tags) if spec.tags else ''} |")
        return 0

    if not specs:
        print("No examples selected.")
        return 0

    results: list[CaseResult] = []
    t0 = time.perf_counter()

    for spec in specs:
        result = _run_case(spec, args)
        results.append(result)

        if result.status == "failed":
            msg = f"[FAILED] {result.relpath} (rc={result.returncode})"
            if result.error:
                msg += f" | {result.error}"
            msg += f" | cmd={result.command}"
            print(msg, file=sys.stderr)
            if args.fail_fast:
                break

    elapsed = time.perf_counter() - t0

    passed = sum(r.status == "passed" for r in results)
    failed = sum(r.status == "failed" for r in results)
    skipped = len(specs) - passed - failed

    print("\n=== Summary ===")
    print(f"selected: {len(specs)}")
    print(f"passed  : {passed}")
    print(f"failed  : {failed}")
    print(f"skipped : {skipped}")
    print(f"elapsed : {elapsed:.2f}s")

    if args.summary_json:
        payload = {
            "root": str(ROOT),
            "selected": len(specs),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "elapsed_seconds": elapsed,
            "results": [r.__dict__ for r in results],
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return 1 if failed > 0 else 0

if __name__ == "__main__":
    raise SystemExit(main())
