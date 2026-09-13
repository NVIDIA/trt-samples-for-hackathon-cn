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

# Convolution "report card": inspect the engine's convolution layers as implicit
# GEMMs (MACs, arithmetic intensity, compute/memory efficiency, M/N/K).
#
# This is the cookbook re-implementation of `trt-engine-explorer`'s
# report_card_convolutions_overview. The original renders an interactive plotly
# dropdown; here each view is a Matplotlib figure saved to a PNG file.

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from tensorrt_cookbook import EnginePlan, annotate_convolutions, case_mark, colors_for, precision_colormap

data_path = Path(__file__).parent.parent / "data"  # shared engine JSON files
out_path = Path(__file__).parent  # this example's own figures
graph_json = data_path / "model.graph.json"
profile_json = data_path / "model.profile.json"

# Truncate long convolution names to this many trailing characters on the axis.
name_tail = 30

def _load_convs():
    plan = EnginePlan(str(graph_json), str(profile_json), name="model")
    return plan, annotate_convolutions(plan)

def _short(name):
    return name if len(name) <= name_tail else "..." + name[-name_tail:]

@case_mark
def case_conv_table():
    """Print the implicit-GEMM metrics of every convolution layer."""
    _, convs = _load_convs()
    if not convs:
        print("No convolution layers in this engine.")
        return
    header = f"{'latency(ms)':>11} {'MACs':>12} {'arith.int':>10} {'M':>7} {'N':>6} {'K':>6}  name"
    print(header)
    print("-" * len(header))
    for c in sorted(convs, key=lambda c: c["latency.avg_time"], reverse=True):
        print(f"{c['latency.avg_time']:>11.4f} {c['attr.macs']:>12d} "
              f"{c['attr.arithmetic_intensity']:>10.2f} {c['attr.M']:>7d} "
              f"{c['attr.N']:>6d} {c['attr.K']:>6d}  {c['Name']}")

@case_mark
def case_conv_metrics():
    """Per-convolution bar charts: latency, MACs, arithmetic intensity, footprint."""
    _, convs = _load_convs()
    if not convs:
        print("No convolution layers in this engine.")
        return
    names = [_short(c["Name"]) for c in convs]
    y = np.arange(len(names))

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(2, 2)

    def barh(ax, values, title, colors="tab:blue"):
        ax.barh(y, values, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title)

    prec_colors = colors_for([c["precision"] for c in convs], precision_colormap)
    barh(fig.add_subplot(gs[0, 0]), [c["latency.avg_time"] for c in convs], "Latency (ms), colored by precision", prec_colors)
    barh(fig.add_subplot(gs[0, 1]), [c["attr.macs"] for c in convs], "Fused Multiply-Accumulates (MACs)")
    barh(fig.add_subplot(gs[1, 0]), [c["attr.arithmetic_intensity"] for c in convs], "Arithmetic intensity (MACs/byte)")
    barh(fig.add_subplot(gs[1, 1]), [c["total_footprint_bytes"] for c in convs], "Data footprint (bytes)")

    fig.suptitle("Convolution characteristics")
    fig.tight_layout()
    out_file = out_path / "conv_metrics.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_conv_roofline():
    """Arithmetic intensity vs compute efficiency (a 2D roofline-style scatter).

    Marker size encodes latency. This is the 2D replacement for the original
    plotly 3D M-N-K scatter (see the GEMM example for the M/N/K view).
    """
    _, convs = _load_convs()
    if not convs:
        print("No convolution layers in this engine.")
        return
    ai = np.array([c["attr.arithmetic_intensity"] for c in convs])
    ce = np.array([c["attr.compute_efficiency"] for c in convs])
    lat = np.array([c["latency.avg_time"] for c in convs])
    sizes = 100 + 3000 * (lat / lat.max() if lat.max() > 0 else lat)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(ai, ce, s=sizes, c=lat, cmap="viridis", alpha=0.7, edgecolors="black")
    for c, x, y in zip(convs, ai, ce):
        ax.annotate(_short(c["Name"]), (x, y), fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Arithmetic intensity (MACs/byte)")
    ax.set_ylabel("Compute efficiency (MACs/ms)")
    ax.set_title("Convolution roofline (marker size / color = latency)")
    fig.colorbar(sc, label="Latency (ms)")

    fig.tight_layout()
    out_file = out_path / "conv_roofline.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    case_conv_table()
    case_conv_metrics()
    case_conv_roofline()

    print("Finish")
