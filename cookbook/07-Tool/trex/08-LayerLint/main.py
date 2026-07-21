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

# Layer linters: heuristics that flag potential performance hazards in an engine.
#
# This is the cookbook re-implementation of `trt-engine-explorer`'s lint.py
# (ConvLinter / ReformatLinter / SliceLinter / QDQLinter). The linters run on the
# engine-graph JSON (no GPU / no profiling data required) and report hazards as
# text; a bar chart summarises the hazard counts per category.

from pathlib import Path

from matplotlib import pyplot as plt

from tensorrt_cookbook import EnginePlan, case_mark, lint_engine

data_path = Path(__file__).parent.parent / "data"  # shared engine JSON files
out_path = Path(__file__).parent  # this example's own figures
graph_json = data_path / "model.graph.json"
profile_json = data_path / "model.profile.json"

def _load():
    return EnginePlan(str(graph_json), str(profile_json), name="model")

@case_mark
def case_lint_report():
    """Run all layer linters and print the hazards they find."""
    plan = _load()
    reports = lint_engine(plan)

    total = sum(len(v) for v in reports.values())
    print(f"Found {total} potential performance hazard(s).\n")
    for category, hazards in reports.items():
        print(f"=== {category} linter: {len(hazards)} hazard(s) ===")
        for h in hazards:
            print(f"  [{h['hazard']}]  {h['name']}")
            for k, v in h.items():
                if k not in ("name", "hazard") and v:
                    print(f"      {k}: {v}")
        print()

@case_mark
def case_lint_summary():
    """Bar chart of the number of hazards found by each linter."""
    plan = _load()
    reports = lint_engine(plan)
    categories = list(reports.keys())
    counts = [len(reports[c]) for c in categories]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(categories, counts, color="tab:red")
    ax.set_ylabel("Number of hazards")
    ax.set_title(f"Layer-lint hazard count by category: {plan.name}")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")

    fig.tight_layout()
    out_file = out_path / "lint_summary.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    case_lint_report()
    case_lint_summary()

    print("Finish")
