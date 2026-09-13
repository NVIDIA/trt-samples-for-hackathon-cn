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

# Consume the JSON that trtexec writes with `--exportProfile` and `--exportTimes`.
#
# main.sh already produces both files but never reads them back, so this script closes the loop:
#   + `--exportProfile` -> per-layer time table, hot-layer ranking, and a two-run A/B comparison.
#   + `--exportTimes`   -> per-iteration H2D / compute / D2H trace plus percentile latency.
#
# This is the cookbook re-implementation of TensorRT-OSS `samples/trtexec/{profiler,tracer}.py`
# (same idea, cookbook style). Re-implementing rather than vendoring also fixes a real
# incompatibility: the upstream tracer expects the pre-TRT-10 key names `startInMs` / `inMs` /
# `outMs`, while current trtexec writes `startH2dMs` / `h2dMs` / `d2hMs`. Both spellings are
# accepted below.

import argparse
import json
import math
from pathlib import Path

from tensorrt_cookbook import case_mark

# Per-layer fields in the `--exportProfile` JSON. `medianMs` appeared in TensorRT-8.6.
PROFILE_FIELDS = ["timeMs", "averageMs", "medianMs", "percentage"]
# Per-iteration interval fields in the `--exportTimes` JSON, current name first, legacy name second.
TRACE_INTERVALS = [("h2dMs", "inMs"), ("computeMs", "computeMs"), ("d2hMs", "outMs"), ("latencyMs", "latencyMs")]

def load_profile(json_file):
    """Split an `--exportProfile` JSON into (iteration count, list of per-layer records)."""
    with open(json_file) as f:
        raw = json.load(f)
    # The first element is the metadata record `{"count": N}`, the rest are layers.
    count = raw[0].get("count", 0) if raw and "count" in raw[0] else 0
    layers = [row for row in raw if "name" in row]
    return count, layers

def load_trace(json_file):
    """Read an `--exportTimes` JSON into a list of per-iteration records."""
    with open(json_file) as f:
        return json.load(f)

def get_interval(row, names):
    """Read one interval from a trace row, tolerating the pre-TRT-10 key spelling."""
    for name in names:
        if name in row:
            return row[name]
    return 0.0

def percentile(sorted_values, ratio):
    """Nearest-rank percentile of an already-sorted list: rank = ceil(ratio * N), 1-based."""
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, math.ceil(ratio * len(sorted_values)) - 1))
    return sorted_values[index]

def print_table(header, rows):
    """Print a list of rows as an aligned text table."""
    width = [len(str(x)) for x in header]
    for row in rows:
        for i, cell in enumerate(row):
            width[i] = max(width[i], len(str(cell)))
    line = " | ".join(f"{str(h):>{w}}" for h, w in zip(header, width))
    print(line)
    print("-" * len(line))
    for row in rows:
        print(" | ".join(f"{str(c):>{w}}" for c, w in zip(row, width)))

@case_mark
def case_profile(profile_json, top_n):
    """Per-layer time table from `--exportProfile`, ranked by total time."""
    count, layers = load_profile(profile_json)
    print(f"Iteration count = {count}, layer count = {len(layers)}")

    layers = sorted(layers, key=lambda row: row.get("timeMs", 0.0), reverse=True)
    rows = []
    for row in layers[:top_n]:
        rows.append([row["name"]] + [f"{row.get(f, 0.0):.6g}" for f in PROFILE_FIELDS])
    # Total is computed over *all* layers, not just the ones displayed.
    total = ["total (all layers)"]
    for f in PROFILE_FIELDS:
        total.append(f"{sum(row.get(f, 0.0) for row in layers):.6g}")
    rows.append(total)

    print(f"Top {min(top_n, len(layers))} layers by total time:")
    print_table(["name"] + PROFILE_FIELDS, rows)

@case_mark
def case_profile_compare(profile_json, reference_json, threshold):
    """A/B two `--exportProfile` runs, reporting per-layer percentage difference."""
    _, layers = load_profile(profile_json)
    _, reference_layers = load_profile(reference_json)
    reference_map = {row["name"]: row for row in reference_layers}

    rows = []
    for row in layers:
        reference = reference_map.pop(row["name"], None)
        if reference is None:  # Layer only exists in the target run (different fusion / tactic).
            rows.append([row["name"], "-", f"{row.get('averageMs', 0.0):.6g}", "new"])
            continue
        reference_average = reference.get("averageMs", 0.0)
        average = row.get("averageMs", 0.0)
        # A zero-time reference layer gives no meaningful ratio, so report it as unavailable.
        difference = (average / reference_average - 1) * 100 if reference_average else float("inf")
        if abs(difference) >= threshold:
            rows.append([row["name"], f"{reference_average:.6g}", f"{average:.6g}", f"{difference:+.2f}%"])
    for name in reference_map:  # Layers that disappeared in the target run.
        rows.append([name, f"{reference_map[name].get('averageMs', 0.0):.6g}", "-", "gone"])

    print(f"Layers differing by >= {threshold}% in averageMs (reference = {Path(reference_json).name}):")
    if rows:
        print_table(["name", "refAverageMs", "averageMs", "% difference"], rows)
    else:
        print("    (none)")

@case_mark
def case_trace(times_json, head_n):
    """Per-iteration trace and percentile latency from `--exportTimes`."""
    trace = load_trace(times_json)
    print(f"Iteration count = {len(trace)}")

    header = [names[0] for names in TRACE_INTERVALS]
    rows = []
    for row in trace[:head_n]:
        rows.append([f"{get_interval(row, names):.6g}" for names in TRACE_INTERVALS])
    print(f"First {len(rows)} iterations (ms):")
    print_table(header, rows)

    # trtexec prints these in its own summary, but only for the run it just did; recomputing them
    # here is what lets us compare saved runs offline.
    latency = sorted(get_interval(row, ("latencyMs", "latencyMs")) for row in trace)
    print("Latency percentiles (ms):")
    rows = [[
        f"{latency[0]:.6g}",
        f"{sum(latency) / len(latency):.6g}",
        f"{percentile(latency, 0.50):.6g}",
        f"{percentile(latency, 0.90):.6g}",
        f"{percentile(latency, 0.95):.6g}",
        f"{percentile(latency, 0.99):.6g}",
        f"{latency[-1]:.6g}",
    ]]
    print_table(["min", "mean", "P50", "P90", "P95", "P99", "max"], rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="model-trained-exportProfile.json", help="`--exportProfile` JSON")
    parser.add_argument("--times", default="model-trained-exportTimes.json", help="`--exportTimes` JSON")
    parser.add_argument("--reference", default=None, help="Second `--exportProfile` JSON to compare against")
    parser.add_argument("--top", type=int, default=10, help="Number of hottest layers to print")
    parser.add_argument("--head", type=int, default=5, help="Number of trace iterations to print")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum %% difference to report")
    args = parser.parse_args()

    case_profile(args.profile, args.top)
    if args.reference is not None:
        case_profile_compare(args.profile, args.reference, args.threshold)
    case_trace(args.times, args.head)

    print("Finish")
