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

# GEMM "report card": inspect the engine's convolutions expressed as matrix
# multiplies (M, N, K) and relate those dimensions to latency.
#
# This is the cookbook re-implementation of `trt-engine-explorer`'s
# report_card_gemm_MNK / report_card_gemm_MNK_scatter / report_card_perf_scatter.
# The originals render interactive plotly 3D scatters; here the 3D M-N-K view is
# projected to 2D Matplotlib figures saved to PNG files.

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from tensorrt_cookbook import EnginePlan, annotate_convolutions, case_mark

data_path = Path(__file__).parent.parent / "data"  # shared engine JSON files
out_path = Path(__file__).parent  # this example's own figures
graph_json = data_path / "model.graph.json"
profile_json = data_path / "model.profile.json"

name_tail = 24  # truncate long convolution names on the axis/labels

def _load_convs():
    plan = EnginePlan(str(graph_json), str(profile_json), name="model")
    return annotate_convolutions(plan)

def _short(name):
    return name if len(name) <= name_tail else "..." + name[-name_tail:]

@case_mark
def case_mnk_bars():
    """Grouped bar chart of the equivalent GEMM dimensions M, N, K per convolution."""
    convs = _load_convs()
    if not convs:
        print("No convolution layers in this engine.")
        return
    names = [_short(c["Name"]) for c in convs]
    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, [c["attr.M"] for c in convs], width, label="M = N*P*Q")
    ax.bar(x, [c["attr.N"] for c in convs], width, label="N = K_out")
    ax.bar(x + width, [c["attr.K"] for c in convs], width, label="K = C*R*S")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Dimension size")
    ax.set_yscale("log")  # M/N/K often span orders of magnitude
    ax.set_title("Convolution as implicit GEMM: M, N, K dimensions")
    ax.legend()

    fig.tight_layout()
    out_file = out_path / "gemm_mnk_bars.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_mnk_vs_latency():
    """M / N / K each vs latency, colored by data footprint (3-panel scatter).

    Direct 2D port of the original plotly `report_card_gemm_MNK_scatter`.
    """
    convs = _load_convs()
    if not convs:
        print("No convolution layers in this engine.")
        return
    lat = np.array([c["latency.avg_time"] for c in convs])
    footprint = np.array([c["total_footprint_bytes"] for c in convs])

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3)
    for i, dim in enumerate(("attr.M", "attr.N", "attr.K")):
        ax = fig.add_subplot(gs[0, i])
        vals = np.array([c[dim] for c in convs])
        sc = ax.scatter(vals, lat, c=footprint, cmap="viridis", s=120, edgecolors="black", alpha=0.8)
        ax.set_xlabel(dim.replace("attr.", ""))
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{dim.replace('attr.', '')} vs latency")
        fig.colorbar(sc, ax=ax, label="footprint (B)")

    fig.suptitle("GEMM dimensions vs latency (color = footprint)")
    fig.tight_layout()
    out_file = out_path / "gemm_mnk_vs_latency.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

@case_mark
def case_mnk_projection():
    """M vs N scatter with marker size = K and color = latency.

    This is the 2D projection of the original plotly 3D M-N-K scatter.
    """
    convs = _load_convs()
    if not convs:
        print("No convolution layers in this engine.")
        return
    M = np.array([c["attr.M"] for c in convs])
    N = np.array([c["attr.N"] for c in convs])
    K = np.array([c["attr.K"] for c in convs])
    lat = np.array([c["latency.avg_time"] for c in convs])
    sizes = 100 + 2000 * (K / K.max() if K.max() > 0 else K)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(M, N, s=sizes, c=lat, cmap="plasma", alpha=0.75, edgecolors="black")
    for c, x, y in zip(convs, M, N):
        ax.annotate(_short(c["Name"]), (x, y), fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("M = N*P*Q")
    ax.set_ylabel("N = K_out")
    ax.set_title("GEMM M x N (marker size = K, color = latency)")
    fig.colorbar(sc, label="Latency (ms)")

    fig.tight_layout()
    out_file = out_path / "gemm_mnk_projection.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    case_mnk_bars()
    case_mnk_vs_latency()
    case_mnk_projection()

    print("Finish")
