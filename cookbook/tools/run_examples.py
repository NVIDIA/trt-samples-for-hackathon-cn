#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified example runner for cookbook.

Goals:
1) Keep each example independently runnable (e.g. `python3 main.py`).
2) Centralize test orchestration to avoid duplicated `unit_test.sh` scripts.
3) Support gradual migration by allowing per-directory `unit_test.yaml` overrides.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "include",
    "LocalFile",
    "tensorrt_cookbook",
    "tensorrt_cookbook.egg-info",
    "tests",
}

DEFAULT_SKIP_CONFIG_FILE = "tools/skip_examples.yaml"
ROOT = Path(__file__).resolve().parents[1]

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

def _normalize_rel(path: Path, base_dir: Path) -> str:
    return path.relative_to(base_dir).as_posix()

def _path_match(relpath: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return True
    p = PurePosixPath(relpath)
    return any(p.match(pattern) for pattern in patterns)

def _discover_examples(base_dir: Path, skip_patterns: list[str] | None = None) -> list[Path]:
    skip_patterns = skip_patterns or []
    candidates: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base_dir):
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_SKIP_DIRS and not d.startswith(".")]

        current = Path(dirpath)
        relpath = _normalize_rel(current, base_dir)

        if relpath != "." and _path_match(relpath, skip_patterns):
            dirnames[:] = []
            continue

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
            run=["python3 main.py > log-main.py.log"],
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

def _run_case(spec: ExampleSpec, args: argparse.Namespace, root_env: dict[str, str]) -> CaseResult:
    start = time.perf_counter()
    merged_env = dict(root_env)
    merged_env.update(spec.env)

    if args.clean:  # TODO: remove this later
        merged_env["TRT_COOKBOOK_CLEAN"] = "1"

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

def _filter_specs(specs: list[ExampleSpec], args: argparse.Namespace) -> list[ExampleSpec]:
    selected: list[ExampleSpec] = []
    for spec in specs:
        if not spec.enabled:
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
    parser.add_argument("--include", action="append", default=["**"], help="glob include pattern on relative path")
    parser.add_argument("--exclude", action="append", default=[], help="glob exclude pattern on relative path")

    parser.add_argument("--tags", action="append", default=[], help="only run cases with any of these tags")
    parser.add_argument("--exclude-tags", action="append", default=[], help="skip cases with these tags")

    parser.add_argument("--timeout", type=int, default=1800, help="default timeout (seconds) per command")
    parser.add_argument("--fail-fast", action="store_true", help="stop at first failed case")
    parser.add_argument("--clean", action="store_true", help="set TRT_COOKBOOK_CLEAN=1 and run clean commands")

    parser.add_argument("--list", action="store_true", help="list discovered cases and exit")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument("--summary-json", type=Path, help="write summary report as JSON")

    args = parser.parse_args()
    if not ROOT.exists():
        print(f"[ERROR] root does not exist: {ROOT}", file=sys.stderr)
        return 2

    os.environ.setdefault("TRT_COOKBOOK_PATH", str(ROOT))
    root_env = dict(os.environ)

    # Skip patterns
    skip_patterns = []
    skip_config = ROOT / DEFAULT_SKIP_CONFIG_FILE
    if skip_config.exists():
        cfg = _load_yaml(skip_config)
        version = cfg.get("version", 1)
        if version != 1:
            raise ValueError(f"{skip_config}: unsupported version {version}, expected 1")
        patterns = _to_list(cfg.get("skip", []), "skip", skip_config)
        skip_patterns = [p.strip() for p in patterns if p.strip()]

    # Get candidates
    candidates: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_SKIP_DIRS and not d.startswith(".")]

        current = Path(dirpath)
        relpath = _normalize_rel(current, ROOT)

        if relpath != "." and _path_match(relpath, skip_patterns):
            dirnames[:] = []
            continue

        has_main = "main.py" in filenames
        has_yaml = "unit_test.yaml" in filenames
        has_skip = ".skip_unit_test" in filenames
        if (has_main or has_yaml) and not has_skip:  # Basic condition to enable unit test
            candidates.append(current)

    candidates.sort()

    specs: list[ExampleSpec] = []
    for c in candidates:
        try:
            spec = _build_spec(c, ROOT, args.timeout)
            if spec is not None:
                specs.append(spec)
        except Exception as e:
            print(f"[ERROR] {c}: {e}", file=sys.stderr)
            return 2

    specs = _filter_specs(specs, args)

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
        result = _run_case(spec, args, root_env)
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
    skipped = len(specs) - len(results)

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
"""
# unit_test.yaml 最小可落地规范（v1）

> 目标：
> - 保持示例目录可单独运行
> - 将测试编排集中到 `tools/run_examples.py`
> - 支持渐进迁移，先跑起来，再逐步替换旧 `unit_test.sh`

## 1) 文件位置

每个示例目录可选放置一个 `unit_test.yaml`：

- `cookbook/02-API/Layer/Cast/unit_test.yaml`
- `cookbook/05-Plugin/BasicExample/unit_test.yaml`

如果目录没有 `unit_test.yaml` 但有 `main.py`，runner 会使用默认行为：

- `run = ["python3 main.py > log-main.py.log"]`

集中跳过目录可在 `tools/skip_examples.yaml` 配置：

```yaml
version: 1
skip:
  - 07-Tool/NetworkSerialization/TRT-8-version
  - 09-TRTLLM/**
```

说明：

- `skip` 支持路径与 glob（语法与 `--include/--exclude` 一致）
- 该文件用于“全局跳过清单”；单目录临时跳过仍可使用 `.skip_unit_test`

## 2) 字段定义（v1）

```yaml
version: 1                # 可选，默认 1
name: Cast / simple       # 可选，展示名
enabled: true             # 可选，默认 true

# 过滤字段
tags: [api, layer, cast]  # 可选，字符串数组

# 执行控制
timeout: 1200             # 可选，单条命令超时（秒）
env:                       # 可选，注入环境变量
  MY_FLAG: "1"

# 生命周期命令（字符串或字符串数组）
pre:                       # 可选，run 前执行
  - "make test"
run:                       # 必填（在使用 unit_test.yaml 时）
  - "python3 main.py > log-main.py.log"
post:                      # 可选，run 后执行
  - "python3 verify.py"

# --clean 时追加执行
clean:                     # 可选
  - "rm -rf *.log"
```

约束：

- `run` 在存在 `unit_test.yaml` 时必须非空
- `pre/run/post/clean` 支持：
  - 单个字符串
  - 字符串数组
- `env` 的 value 支持标量（string/int/float/bool），会转换成字符串

## 3) 推荐迁移策略

1. 先引入 runner，不改示例代码
2. 对“特殊目录”补 `unit_test.yaml`（如需要 `make test`、`main.sh`）
3. 普通目录不写配置，直接走默认 `main.py` 规则
4. 最后把旧 `unit_test.sh` 改成薄壳或移除

## 4) 参数接口（run_examples.py）

基础：

- 根目录固定为脚本所在 `cookbook` 路径（`ROOT = Path(__file__).resolve().parents[1]`）
- `--list`：只列出可运行示例
- `--dry-run`：只打印命令，不执行
- `--summary-json PATH`：输出 JSON 汇总报告

选择：

- `--case REL_PATH`：精确运行某个相对路径（可重复）
- `--include GLOB`：按 glob 包含（可重复，默认 `**`）
- `--exclude GLOB`：按 glob 排除（可重复）
- `--tags TAG`：只运行包含任一 tag 的示例（可重复）
- `--exclude-tags TAG`：排除带指定 tag 的示例（可重复）
- `--gpu-only`：只运行 `requires_gpu=true` 的示例

执行策略：

- `--timeout SEC`：默认每条命令超时（默认 1800）
- `--fail-fast`：首个失败即停止
- `--clean`：注入 `TRT_COOKBOOK_CLEAN=1`，并执行 `clean`

## 5) 配置样例

### 5.1 普通目录（可不写）

无 `unit_test.yaml`，目录内有 `main.py` 即可。

### 5.2 需要额外步骤的目录

```yaml
version: 1
tags: [plugin, compile]
pre:
  - "make test"
run:
  - "python3 main.py > log-main.py.log"
clean:
  - "make clean"
  - "rm -rf *.log"
```

### 5.3 非 main.py 入口目录

```yaml
version: 1
tags: [tool, polygraphy]
run:
  - "chmod +x main.sh"
  - "./main.sh"
  - "polygraphy run --help > Help-run.txt"
clean:
  - "rm -rf *.json *.lock *.log *.onnx *.so *.TimingCache *.trt polygraphy_run.py"
```
"""
