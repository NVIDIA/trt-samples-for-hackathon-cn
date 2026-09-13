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

# Latency "report card": inspect where an engine spends its time.
#
# This is the cookbook re-implementation of `trt-engine-explorer`'s
# report_card_perf_overview / layer_latency_sunburst / plot_engine_timings.
# The original renders an interactive plotly dropdown of ~10 views; here each
# view is a Matplotlib figure saved to a PNG file (no notebook / browser / hover).

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from tensorrt_cookbook import (
    EnginePlan,
    case_mark,
    colors_for,
    group_count,
    group_sum,
    layer_colormap,
    precision_colormap,
    read_timing_file,
)

data_path = Path(__file__).parent.parent / "data"  # shared engine JSON files
out_path = Path(__file__).parent  # this example's own figures
graph_json = data_path / "model.graph.json"
profile_json = data_path / "model.profile.json"
timing_json = data_path / "model.timing.json"

# Plotting range/size knobs (the plotly Range-Slider is intentionally dropped;
# adjust these and re-run to change what is drawn).
n_top_layers = 20  # how many individual layers to show in per-layer charts
hist_bins = 20  # number of bins in the latency-distribution histogram

def _load():
    return EnginePlan(str(graph_json), str(profile_json), name="model")

@case_mark
def case_latency_by_type():
    """Latency aggregated per layer type, in milliseconds and in percent."""
    plan = _load()
    lat_ms = group_sum(plan.records, "type", "latency.avg_time")
    lat_pct = group_sum(plan.records, "type", "latency.pct_time")
    counts = group_count(plan.records, "type")

    types = sorted(lat_ms, key=lat_ms.get)  # ascending for a horizontal bar
    colors = colors_for(types, layer_colormap)

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.barh(types, [lat_ms[t] for t in types], color=colors)
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Latency per layer type (ms)")

    ax = fig.add_subplot(gs[0, 1])
    ax.barh(types, [lat_pct[t] for t in types], color=colors)
    ax.set_xlabel("Latency (%)")
    ax.set_title("Latency per layer type (%)")

    ax = fig.add_subplot(gs[0, 2])
    ax.barh(types, [counts[t] for t in types], color=colors)
    ax.set_xlabel("Count")
    ax.set_title("Layer count per type")

    fig.suptitle(f"Latency by layer type: {plan.name}")
    fig.tight_layout()
    out_file = out_path / "latency_by_type.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_latency_per_layer():
    """Per-layer latency (ms), colored by layer type - the slowest N layers."""
    plan = _load()
    names = plan.col("Name").tolist()
    types = plan.col("type").tolist()
    avg = plan.col("latency.avg_time").astype(float)

    order = np.argsort(avg)[::-1][:n_top_layers]
    sel_names = [names[i] for i in order]
    sel_types = [types[i] for i in order]
    sel_avg = [avg[i] for i in order]
    colors = colors_for(sel_types, layer_colormap)

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(sel_names))
    ax.barh(y, sel_avg, color=colors)
    ax.set_yticks(y)
    # Long layer names: keep only the tail for readability.
    ax.set_yticklabels([n if len(n) < 40 else "..." + n[-37:] for n in sel_names], fontsize=7)
    ax.invert_yaxis()  # slowest on top
    ax.set_xlabel("Latency (ms)")
    ax.set_title(f"Top {n_top_layers} slowest layers (colored by type): {plan.name}")

    # A manual legend keyed on the layer types actually present.
    present = list(dict.fromkeys(sel_types))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors_for(present, layer_colormap)]
    ax.legend(handles, present, fontsize=8, title="Layer type")

    fig.tight_layout()
    out_file = out_path / "latency_per_layer.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_latency_distribution():
    """Histogram of the per-layer latency-percentage contribution."""
    plan = _load()
    pct = plan.col("latency.pct_time").astype(float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pct, bins=hist_bins, color="tab:blue", edgecolor="black")
    ax.set_xlabel("Per-layer latency (%)")
    ax.set_ylabel("Number of layers")
    ax.set_title(f"Layer latency distribution: {plan.name}")

    fig.tight_layout()
    out_file = out_path / "latency_distribution.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_precision_rollup():
    """Layer count and latency share grouped by precision (pie charts).

    Replaces the plotly precision-sunburst / pie-rollup with two 2D pies.
    """
    plan = _load()
    count_by_prec = group_count(plan.records, "precision")
    lat_by_prec = group_sum(plan.records, "precision", "latency.pct_time")

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(1, 2)

    ax = fig.add_subplot(gs[0, 0])
    keys = list(count_by_prec.keys())
    ax.pie(list(count_by_prec.values()), labels=keys, autopct="%1.0f%%", colors=colors_for(keys, precision_colormap))
    ax.set_title("Layer count by precision")

    ax = fig.add_subplot(gs[0, 1])
    keys = list(lat_by_prec.keys())
    ax.pie(list(lat_by_prec.values()), labels=keys, autopct="%1.1f%%", colors=colors_for(keys, precision_colormap))
    ax.set_title("% latency by precision")

    fig.suptitle(f"Precision rollup: {plan.name}")
    fig.tight_layout()
    out_file = out_path / "precision_rollup.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_latency_by_type_precision():
    """Stacked bar of latency per type, split by precision.

    A flat 2D replacement for the original (type -> latency) sunburst.
    """
    plan = _load()
    types = sorted(set(plan.col("type").tolist()))
    precisions = sorted(set(plan.col("precision").tolist()), key=lambda p: str(p))

    # latency[type][precision] = summed avg latency
    rec_type = plan.col("type").tolist()
    rec_prec = plan.col("precision").tolist()
    rec_lat = plan.col("latency.avg_time").astype(float)
    matrix = {t: {p: 0.0 for p in precisions} for t in types}
    for t, p, l in zip(rec_type, rec_prec, rec_lat):
        matrix[t][p] += l

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(types))
    for p in precisions:
        vals = np.array([matrix[t][p] for t in types])
        ax.bar(types, vals, bottom=bottom, label=str(p), color=precision_colormap.get(p, "gray"))
        bottom += vals
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Latency per layer type, split by precision: {plan.name}")
    ax.legend(title="Precision")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    out_file = out_path / "latency_by_type_precision.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_engine_timings():
    """Per-iteration end-to-end latency samples from the timing JSON."""
    if not timing_json.exists():
        print(f"Timing file {timing_json} not found; skipping.")
        return
    latencies = read_timing_file(str(timing_json))
    samples = np.arange(len(latencies))
    mean = float(np.mean(latencies))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(samples, latencies, s=8, color="tab:blue", alpha=0.6, label="samples")
    ax.axhline(mean, color="red", linestyle="--", label=f"mean = {mean:.4f} ms")
    ax.set_xlabel("Timing sample")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Engine timing samples")
    ax.legend()

    fig.tight_layout()
    out_file = out_path / "engine_timings.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    case_latency_by_type()
    case_latency_per_layer()
    case_latency_distribution()
    case_precision_rollup()
    case_latency_by_type_precision()
    case_engine_timings()

    print("Finish")
