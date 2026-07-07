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

# Precision "report card": how are precisions (INT8 / FP16 / FP32 / ...) used and
# where does the engine reformat/convert data?
#
# This is the cookbook re-implementation of the precision-related views from
# `trt-engine-explorer`'s report_card_perf_overview (precision_per_layer,
# precision_per_type sunburst, precision statistics) plus
# report_card_reformat_overview. The original renders interactive plotly
# dropdowns; here each view is a Matplotlib figure saved to a PNG file.

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from tensorrt_cookbook import (
    EnginePlan,
    case_mark,
    colors_for,
    compute_precision_stats,
    group_count,
    group_sum,
    precision_colormap,
    print_precision_stats,
)

data_path = Path(__file__).parent.parent / "data"  # shared engine JSON files
out_path = Path(__file__).parent  # this example's own figures
graph_json = data_path / "model.graph.json"
profile_json = data_path / "model.profile.json"

n_top_layers = 20  # how many layers to show in per-layer charts

def _load():
    return EnginePlan(str(graph_json), str(profile_json), name="model")

def _short(name, tail=30):
    return name if len(name) <= tail else "..." + name[-tail:]

@case_mark
def case_precision_stats():
    """Print the byte breakdown per precision (activations and weights)."""
    plan = _load()
    print_precision_stats(plan)

@case_mark
def case_precision_per_layer():
    """Per-layer latency colored by the layer's (input) precision."""
    plan = _load()
    names = plan.col("Name").tolist()
    prec = plan.col("precision").tolist()
    avg = plan.col("latency.avg_time").astype(float)

    order = np.argsort(avg)[::-1][:n_top_layers]
    y = np.arange(len(order))
    colors = colors_for([prec[i] for i in order], precision_colormap)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(y, avg[order], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(names[i]) for i in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Latency (ms)")
    ax.set_title(f"Latency per layer, colored by precision: {plan.name}")

    present = list(dict.fromkeys(prec[i] for i in order))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors_for(present, precision_colormap)]
    ax.legend(handles, [str(p) for p in present], title="Precision", fontsize=8)

    fig.tight_layout()
    out_file = out_path / "precision_per_layer.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_precision_stats_bar():
    """Bytes per precision for input activations, output activations and weights."""
    plan = _load()
    stats = compute_precision_stats(plan)

    fig = plt.figure(figsize=(15, 4))
    gs = fig.add_gridspec(1, 3)
    for i, key in enumerate(("input_activations", "output_activations", "weights")):
        ax = fig.add_subplot(gs[0, i])
        data = stats[key]
        keys = list(data.keys())
        vals = np.array([data[k] for k in keys]) / 1024
        ax.bar([str(k) for k in keys], vals, color=colors_for(keys, precision_colormap))
        ax.set_ylabel("Size (KB)")
        ax.set_title(key.replace("_", " "))

    fig.suptitle(f"Byte footprint per precision: {plan.name}")
    fig.tight_layout()
    out_file = out_path / "precision_stats.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_precision_per_type():
    """Layer count per type, split by precision (2D replacement for the sunburst)."""
    plan = _load()
    types = sorted(set(plan.col("type").tolist()))
    precisions = sorted(set(plan.col("precision").tolist()), key=str)

    rec_type = plan.col("type").tolist()
    rec_prec = plan.col("precision").tolist()
    matrix = {t: {p: 0 for p in precisions} for t in types}
    for t, p in zip(rec_type, rec_prec):
        matrix[t][p] += 1

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(types))
    for p in precisions:
        vals = np.array([matrix[t][p] for t in types])
        ax.bar(types, vals, bottom=bottom, label=str(p), color=precision_colormap.get(p, "gray"))
        bottom += vals
    ax.set_ylabel("Layer count")
    ax.set_title(f"Layer count per type, split by precision: {plan.name}")
    ax.legend(title="Precision")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    out_file = out_path / "precision_per_type.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_reformat_overview():
    """How Reformat layers are used, grouped by their origin (e.g. QDQ, ...).

    Port of `report_card_reformat_overview`. Reformat layers convert tensor
    layout/precision; grouping by origin shows why they were inserted.
    """
    plan = _load()
    reformats = plan.get_layers_by_type("Reformat")
    if not reformats:
        print("The engine plan does not contain Reformat layers.")
        return
    count_by_origin = group_count(reformats, "Origin")
    pct_by_origin = group_sum(reformats, "Origin", "latency.pct_time")

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(1, 2)

    ax = fig.add_subplot(gs[0, 0])
    origins = list(count_by_origin.keys())
    ax.barh(origins, [count_by_origin[o] for o in origins], color="tab:cyan")
    ax.set_xlabel("Count")
    ax.set_title("Reformat count by origin")

    ax = fig.add_subplot(gs[0, 1])
    origins = list(pct_by_origin.keys())
    ax.barh(origins, [pct_by_origin[o] for o in origins], color="tab:cyan")
    ax.set_xlabel("Latency (%)")
    ax.set_title("Reformat % latency by origin")

    fig.suptitle(f"Reformat overview: {plan.name}")
    fig.tight_layout()
    out_file = out_path / "reformat_overview.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    case_precision_stats()
    case_precision_per_layer()
    case_precision_stats_bar()
    case_precision_per_type()
    case_reformat_overview()

    print("Finish")
