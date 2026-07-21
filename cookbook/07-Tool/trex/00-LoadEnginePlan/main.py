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

# Load a TensorRT engine plan and its profiling data, then summarise the engine
# structure and performance as text + a Matplotlib figure.
#
# This is the cookbook re-implementation of `trt-engine-explorer`'s
# EnginePlan / print_summary / precision-stats API, without pandas or plotly.

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from tensorrt_cookbook import (
    EnginePlan,
    case_mark,
    compute_precision_stats,
    group_sum,
    print_precision_stats,
    print_summary,
)

data_path = Path(__file__).parent.parent / "data"  # shared engine JSON files
out_path = Path(__file__).parent  # this example's own figures
graph_json = data_path / "model.graph.json"
profile_json = data_path / "model.profile.json"

# Number of slowest layers to list in the text report (adjust and re-run to taste).
n_top_layers = 5

@case_mark
def case_load_and_summarize():
    # Load graph + profiling JSON into an EnginePlan (no GPU required).
    plan = EnginePlan(str(graph_json), str(profile_json), name="model")

    print(f"Engine plan name = {plan.name}")
    print(f"Number of layers = {len(plan.records)}")
    print(f"Total runtime    = {plan.total_runtime:.4f} ms")
    print()

    # Textual engine summary (inputs/outputs, weights/activation footprint, ...).
    print_summary(plan)

    # Per-precision byte breakdown of activations and weights.
    print_precision_stats(plan)

@case_mark
def case_layer_report():
    plan = EnginePlan(str(graph_json), str(profile_json), name="model")

    # Latency aggregated per layer type (a pandas-free groupby-sum).
    latency_by_type = group_sum(plan.records, "type", "latency.avg_time")
    print("Latency (ms) by layer type:")
    for layer_type, latency in sorted(latency_by_type.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {layer_type:<16s}: {latency:.4f}")

    # The slowest individual layers.
    names = plan.col("Name")
    avg = plan.col("latency.avg_time").astype(float)
    order = np.argsort(avg)[::-1][:n_top_layers]
    print(f"\nTop {n_top_layers} slowest layers:")
    for i in order:
        print(f"    {avg[i]:.4f} ms  [{plan.records[i]['type']}]  {names[i]}")

@case_mark
def case_plot():
    plan = EnginePlan(str(graph_json), str(profile_json), name="model")

    # Figure 1: two panels laid out with GridSpec (replaces plotly subplots).
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2)

    # Panel A: latency (ms) per layer type, as a horizontal bar chart.
    latency_by_type = group_sum(plan.records, "type", "latency.avg_time")
    items = sorted(latency_by_type.items(), key=lambda kv: kv[1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.barh([k for k, _ in items], [v for _, v in items], color="tab:blue")
    ax0.set_xlabel("Latency (ms)")
    ax0.set_title("Latency by layer type")

    # Panel B: weights byte footprint per precision, as a pie chart.
    weights = compute_precision_stats(plan)["weights"]
    ax1 = fig.add_subplot(gs[0, 1])
    if weights:
        ax1.pie(list(weights.values()), labels=list(weights.keys()), autopct="%1.1f%%")
    ax1.set_title("Weights footprint by precision")

    fig.suptitle(f"Engine plan overview: {plan.name}")
    fig.tight_layout()
    out_file = out_path / "overview.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved figure to {out_file}")

if __name__ == "__main__":
    case_load_and_summarize()
    case_layer_report()
    case_plot()

    print("Finish")
