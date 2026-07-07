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

# Memory "report card": where does the engine spend bytes (weights vs activations)?
#
# This is the cookbook re-implementation of `trt-engine-explorer`'s
# report_card_memory_footprint. The original renders an interactive plotly
# dropdown; here each view is a Matplotlib figure saved to a PNG file.
#
# Per-layer columns used (already computed by EnginePlan):
#   weights_size          - bytes of the layer's constant weights
#   total_io_size_bytes   - bytes of the layer's input + output activations
#   total_footprint_bytes - weights + activations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from tensorrt_cookbook import EnginePlan, case_mark, group_sum

data_path = Path(__file__).parent.parent / "data"  # shared engine JSON files
out_path = Path(__file__).parent  # this example's own figures
graph_json = data_path / "model.graph.json"
profile_json = data_path / "model.profile.json"

n_top_layers = 20  # how many layers to show in per-layer charts
hist_bins = 20  # number of bins in the footprint-distribution histogram

def _load():
    return EnginePlan(str(graph_json), str(profile_json), name="model")

def _short(name, tail=30):
    return name if len(name) <= tail else "..." + name[-tail:]

@case_mark
def case_memory_table():
    """Print the total footprint and the layers with the largest footprint."""
    plan = _load()
    mb = 1024 * 1024
    print(f"Total weights     = {plan.total_weights_size / mb:.3f} MB")
    print(f"Total activations = {plan.total_act_size / mb:.3f} MB")
    total = plan.total_weights_size + plan.total_act_size
    print(f"Total footprint   = {total / mb:.3f} MB")

    names = plan.col("Name").tolist()
    footprint = plan.col("total_footprint_bytes").astype(float)
    order = np.argsort(footprint)[::-1][:n_top_layers]
    print(f"\nTop {n_top_layers} layers by footprint:")
    for i in order:
        print(f"    {footprint[i] / 1024:>10.1f} KB  [{plan.records[i]['type']}]  {names[i]}")

@case_mark
def case_footprint_per_layer():
    """Stacked bar of weights + activations bytes for the largest-footprint layers."""
    plan = _load()
    names = plan.col("Name").tolist()
    weights = plan.col("weights_size").astype(float)
    acts = plan.col("total_io_size_bytes").astype(float)
    footprint = plan.col("total_footprint_bytes").astype(float)

    order = np.argsort(footprint)[::-1][:n_top_layers]
    y = np.arange(len(order))
    sel_names = [_short(names[i]) for i in order]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(y, weights[order] / 1024, color="tab:orange", label="weights")
    ax.barh(y, acts[order] / 1024, left=weights[order] / 1024, color="tab:blue", label="activations")
    ax.set_yticks(y)
    ax.set_yticklabels(sel_names, fontsize=7)
    ax.invert_yaxis()  # largest on top
    ax.set_xlabel("Footprint (KB)")
    ax.set_title(f"Top {n_top_layers} layers by memory footprint: {plan.name}")
    ax.legend()

    fig.tight_layout()
    out_file = out_path / "footprint_per_layer.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_footprint_by_type():
    """Weights and activation bytes aggregated per layer type (stacked bar)."""
    plan = _load()
    types = sorted(set(plan.col("type").tolist()))
    w_by_type = group_sum(plan.records, "type", "weights_size")
    a_by_type = group_sum(plan.records, "type", "total_io_size_bytes")

    w = np.array([w_by_type[t] for t in types]) / 1024
    a = np.array([a_by_type[t] for t in types]) / 1024

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(types, w, color="tab:orange", label="weights")
    ax.bar(types, a, bottom=w, color="tab:blue", label="activations")
    ax.set_ylabel("Footprint (KB)")
    ax.set_title(f"Memory footprint per layer type: {plan.name}")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    out_file = out_path / "footprint_by_type.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_footprint_distribution():
    """Histograms of the per-layer weights / activation / total footprint."""
    plan = _load()
    weights = plan.col("weights_size").astype(float) / 1024
    acts = plan.col("total_io_size_bytes").astype(float) / 1024
    total = plan.col("total_footprint_bytes").astype(float) / 1024

    fig = plt.figure(figsize=(15, 4))
    gs = fig.add_gridspec(1, 3)
    for i, (data, title) in enumerate((
        (weights, "Weights"),
        (acts, "Activations"),
        (total, "Total footprint"),
    )):
        ax = fig.add_subplot(gs[0, i])
        ax.hist(data, bins=hist_bins, color="tab:blue", edgecolor="black")
        ax.set_xlabel("Size (KB)")
        ax.set_ylabel("Number of layers")
        ax.set_title(title)

    fig.suptitle(f"Per-layer footprint distribution: {plan.name}")
    fig.tight_layout()
    out_file = out_path / "footprint_distribution.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    case_memory_table()
    case_footprint_per_layer()
    case_footprint_by_type()
    case_footprint_distribution()

    print("Finish")
