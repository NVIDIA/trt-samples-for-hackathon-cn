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
"""Compare two benchmark CSVs without claiming a hardware root cause."""
import argparse
import csv
import math
from pathlib import Path

def load(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"gpu", "driver_version", "runtime_version", "iters", "warmup", "rounds", "test", "status", "metric", "unit", "samples", "mean_ns", "note"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing CSV columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    indexed = {}
    metadata_fields = ("gpu", "driver_version", "runtime_version", "iters", "warmup", "rounds")
    metadata = {field: rows[0][field] for field in metadata_fields}
    for line, row in enumerate(rows, 2):
        if None in row or any(row.get(field) is None for field in required):
            raise ValueError(f"{path}:{line}: malformed CSV row")
        if any(row[field] != metadata[field] for field in metadata_fields):
            raise ValueError(f"{path}:{line}: mixed run metadata")
        if row["status"] not in ("OK", "SKIP"):
            raise ValueError(f"{path}:{line}: invalid status")
        key = (row["test"], row["metric"], row["unit"])
        if key in indexed:
            raise ValueError(f"{path}:{line}: duplicate metric {key}")
        if row["status"] == "OK":
            mean = float(row["mean_ns"])
            if not math.isfinite(mean) or mean < 0 or int(row["samples"]) < 1:
                raise ValueError(f"{path}:{line}: invalid measurement")
            row = dict(row, mean=mean)
        indexed[key] = row
    return metadata, indexed

def ratio(numerator, denominator):
    return f"{numerator / denominator:.3f}x" if denominator > 0 else "N/A"

def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")

def find_mean(rows, test, metric_contains=None):
    found = [row["mean"] for row in rows.values() if row["status"] == "OK" and row["test"] == test and (metric_contains is None or metric_contains in row["metric"])]
    return found[0] if len(found) == 1 else None

def compare(baseline_path, candidate_path):
    base_meta, baseline = load(baseline_path)
    cand_meta, candidate = load(candidate_path)
    lines = ["# CUDA launch 敏感性对照", "", f"基准：{cell(base_meta['gpu'])}；对比：{cell(cand_meta['gpu'])}。", "", "倍率 = 对比卡 / 基准卡；大于 1 表示该指标耗时更高。所有时间为 ns。", "这份报告描述观测差异，不证明 I2M、semaphore、QMD 或任何硬件模块是根因。", ""]
    mismatched = [field for field in ("driver_version", "runtime_version", "iters", "warmup", "rounds") if base_meta[field] != cand_meta[field]]
    if mismatched:
        lines += ["配置不一致，跨卡差异存在混杂：" + "; ".join(f"{field}={base_meta[field]} / {cand_meta[field]}" for field in mismatched), ""]
    lines += ["| 测试 | 指标 | 单位 | 基准均值 | 对比均值 | 对比/基准 |", "|---|---|---|---:|---:|---:|"]
    matched = 0
    for key, a in baseline.items():
        b = candidate.get(key)
        if a["status"] != "OK" or not b or b["status"] != "OK":
            continue
        matched += 1
        lines.append(f"| {cell(key[0])} | {cell(key[1])} | {cell(key[2])} | "
                     f"{a['mean']:.3f} | {b['mean']:.3f} | {ratio(b['mean'], a['mean'])} |")
    if not matched:
        raise ValueError("The files have no matching successful metrics")
    lines += ["", "## 同卡内部的敏感性", "", "这些比值比较同卡的实验与对照，再并列观察两张卡。它们不是硬件阶段计时。", "", "| 对照 | 基准卡内部比值 | 对比卡内部比值 |", "|---|---:|---:|"]
    comparisons = [
        ("256 B / 无参数：API", "7b", "7a", None),
        ("1024 B / 无参数：API", "7c", "7a", None),
        ("1024 B / 无参数：端到端每 kernel", "7c.t", "7a.t", None),
        ("单流 / 八流：每 kernel（>1 表示八流更快）", "8a", "8b", None),
        ("48 KiB / 无共享内存：相同 8×32，API", "10c", "10b", None),
        ("48 KiB / 无共享内存：相同 8×32，吞吐", "10c.t", "10b.t", None),
        ("8 blocks / 1 block：均为32线程，API", "10b", "10d", None),
        ("32线程 / 1线程：均为1 block，API", "10d", "10a", None),
    ]
    for label, experiment, control, _ in comparisons:
        values = [find_mean(rows, test) for rows in (baseline, candidate) for test in (experiment, control)]
        if None in values:
            continue
        a, a0, b, b0 = values
        lines.append(f"| {label} | {ratio(a, a0)} | {ratio(b, b0)} |")
    lines += ["", "## Graph 随节点数的变化", "", "| 节点数 | 基准 ns/replay | 对比 ns/replay | 基准摊销 ns/kernel | 对比摊销 ns/kernel |", "|---:|---:|---:|---:|---:|"]
    for nodes in (1, 8, 32):
        needle = f"nodes={nodes}:"
        a = find_mean(baseline, "9b", needle)
        b = find_mean(candidate, "9b", needle)
        if a is not None and b is not None:
            lines.append(f"| {nodes} | {a:.3f} | {b:.3f} | {a/nodes:.3f} | {b/nodes:.3f} |")
    lines += ["", "## 缺失或跳过", ""]
    unavailable = 0
    for label, rows, other in (("基准", baseline, candidate), ("对比", candidate, baseline)):
        for key, row in rows.items():
            if row["status"] == "SKIP":
                lines.append(f"- {label} [{cell(key[0])}] SKIP：{cell(row['note'])}")
                unavailable += 1
            elif key not in other or other[key]["status"] != "OK":
                lines.append(f"- {label} [{cell(key[0])}] {cell(key[1])}：另一文件没有匹配的有效指标。")
                unavailable += 1
    if not unavailable:
        lines.append("无。")
    lines += ["", "## 解读边界", "", "- 参数变大后的增量属于参数敏感性，不能标成 I2M 耗时。", "- 多流改善也可能来自并发和资源调度，不能单独证明 semaphore 慢。", "- Graph 节点增加时实际执行工作也增加；需同时看每 kernel 摊销值和 GPU 时间线。", "- grid/block/smem 会改变配置、调度和背压，不等于 QMD 填充时间。", "- API 与 GPU 可以重叠；不能把不同测试的均值直接相减当作纯 GPU 时间。", "- CSV 没有原始样本和重复独立运行数据，本报告不做统计显著性或根因判定。", "- 吞吐 p99 是轮平均的分位数，不是单个 kernel/launch 的尾延迟。", "- CSV 未记录的 CPU、OS、驱动包、频率和其他负载仍需另行控制。", ""]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline CSV, e.g. L20")
    parser.add_argument("candidate", help="Candidate CSV, e.g. RTX PRO 5000")
    parser.add_argument("--output", type=Path, help="Write Markdown instead of stdout")
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() in (Path(args.baseline).resolve(), Path(args.candidate).resolve()):
            raise ValueError("Output must not overwrite either input CSV")
        report = compare(args.baseline, args.candidate)
        if args.output:
            args.output.write_text(report, encoding="utf-8")
            print(f"Report: {args.output}")
        else:
            print(report)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"ERROR: {error}\n")

if __name__ == "__main__":
    main()
