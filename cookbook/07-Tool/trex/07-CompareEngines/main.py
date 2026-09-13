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

# Compare two (or more) TensorRT engine plans - here an INT8 engine vs an FP16
# engine built from the same MNIST network.
#
# This is the cookbook re-implementation of `trt-engine-explorer`'s
# compare_engines.py (compare_engines_overview / compare_engines_summaries_tbl).
# The original renders interactive plotly dropdowns; here each view is a
# Matplotlib figure saved to a PNG file, plus a printed summary table.

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from tensorrt_cookbook import EnginePlan, case_mark, group_sum, layer_colormap, precision_colormap, summary_dict

data_path = Path(__file__).parent.parent / "data"  # shared engine JSON files
out_path = Path(__file__).parent  # this example's own figures

# The engines to compare: (name, graph JSON, profile JSON). Add more entries to
# compare more than two engines.
engines = [
    ("int8", data_path / "model.graph.json", data_path / "model.profile.json"),
    ("fp16", data_path / "model.fp16.graph.json", data_path / "model.fp16.profile.json"),
]

def _load_plans():
    return [EnginePlan(str(g), str(p), name=name) for name, g, p in engines]

@case_mark
def case_summary_table():
    """Print a side-by-side summary of the engines and the overall speedup."""
    plans = _load_plans()
    summaries = [summary_dict(p) for p in plans]
    names = [p.name for p in plans]

    keys = ["Layers", "Average time", "Weights", "Activations"]
    width = max(len(k) for k in keys) + 2
    print(f"{'attribute':<{width}}" + "".join(f"{n:>16}" for n in names))
    print("-" * (width + 16 * len(names)))
    for k in keys:
        print(f"{k:<{width}}" + "".join(f"{s.get(k, ''):>16}" for s in summaries))

    # Speedup of the last engine relative to the first (sum-of-layer-latencies).
    t0, t1 = plans[0].total_runtime, plans[-1].total_runtime
    print(f"\nTotal runtime: {names[0]} = {t0:.4f} ms, {names[-1]} = {t1:.4f} ms")
    if t1 > 0:
        print(f"Speedup ({names[0]} vs {names[-1]}) = {t1 / t0:.3f}x")

@case_mark
def case_latency_by_type():
    """Stacked bar of latency-per-type for each engine, side by side."""
    plans = _load_plans()
    names = [p.name for p in plans]
    # Union of all layer types across engines (stable order).
    types = sorted({t for p in plans for t in p.col("type").tolist()})
    lat = [group_sum(p.records, "type", "latency.avg_time") for p in plans]

    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(plans))
    x = np.arange(len(plans))
    for t in types:
        vals = np.array([lat[i].get(t, 0) for i in range(len(plans))])
        ax.bar(x, vals, bottom=bottom, label=t, color=layer_colormap.get(t, "gray"))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency by layer type, per engine")
    ax.legend(title="Layer type", fontsize=8)

    fig.tight_layout()
    out_file = out_path / "compare_latency_by_type.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_latency_by_type_grouped():
    """Grouped bar comparing per-type latency between engines (+ speedup table)."""
    plans = _load_plans()
    names = [p.name for p in plans]
    types = sorted({t for p in plans for t in p.col("type").tolist()})
    lat = [group_sum(p.records, "type", "latency.avg_time") for p in plans]

    y = np.arange(len(types))
    n = len(plans)
    height = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, p in enumerate(plans):
        vals = [lat[i].get(t, 0) for t in types]
        ax.barh(y + (i - (n - 1) / 2) * height, vals, height, label=names[i])
    ax.set_yticks(y)
    ax.set_yticklabels(types)
    ax.invert_yaxis()
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Latency by layer type: engine comparison")
    ax.legend(title="Engine")

    fig.tight_layout()
    out_file = out_path / "compare_latency_by_type_grouped.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

    # Per-type speedup of the last engine relative to the first.
    print(f"\nPer-type speedup ({names[0]} vs {names[-1]}):")
    for t in types:
        a, b = lat[0].get(t, 0), lat[-1].get(t, 0)
        speedup = f"{b / a:.2f}x" if a > 0 else "n/a"
        print(f"    {t:<16}: {names[0]}={a:.4f}  {names[-1]}={b:.4f}  speedup={speedup}")

@case_mark
def case_latency_by_precision():
    """Stacked bar of latency-per-precision for each engine, side by side."""
    plans = _load_plans()
    names = [p.name for p in plans]
    precisions = sorted({str(pr) for p in plans for pr in p.col("precision").tolist()})
    lat = [group_sum(p.records, "precision", "latency.avg_time") for p in plans]

    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(plans))
    x = np.arange(len(plans))
    for pr in precisions:
        vals = np.array([lat[i].get(pr, 0) for i in range(len(plans))])
        ax.bar(x, vals, bottom=bottom, label=pr, color=precision_colormap.get(pr, "gray"))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency by precision, per engine")
    ax.legend(title="Precision")

    fig.tight_layout()
    out_file = out_path / "compare_latency_by_precision.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    case_summary_table()
    case_latency_by_type()
    case_latency_by_type_grouped()
    case_latency_by_precision()

    print("Finish")
