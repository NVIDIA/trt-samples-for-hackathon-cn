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

# Summarize an engine by tactic, and generate a consolidated "engine card".
#
# This ports the two summary utilities added on the trex main / 10.9 branches:
#   - `utils/summarize_engine.py` -> a per-tactic latency table (`summary` subcmd)
#   - `utils/gen_engine_card.py`  -> an engine "report card"
#
# The original engine card is a browser-opening HTML page; following the cookbook
# style (text/figures to file, no browser/interactivity) it is produced here as a
# Markdown file that bundles the summary, layer-type/tactic/precision breakdowns
# and the lint hazards.

from pathlib import Path

from tabulate import tabulate

from tensorrt_cookbook import (
    EnginePlan,
    case_mark,
    compute_precision_stats,
    group_count,
    group_sum,
    lint_engine,
    summarize_engine_tactics,
    summary_dict,
)

data_path = Path(__file__).parent.parent / "data"  # shared engine JSON files
out_path = Path(__file__).parent
graph_json = data_path / "model.graph.json"
profile_json = data_path / "model.profile.json"
card_file = out_path / "engine_card.md"

n_top_tactics = 10  # how many tactics to list in the card

def _load():
    return EnginePlan(str(graph_json), str(profile_json), name="model")

@case_mark
def case_summarize_tactics():
    """Print a per-tactic latency summary (the trex `summary` sub-command)."""
    plan = _load()
    rows = summarize_engine_tactics(plan, group_tactics=True, sort_key="latency")
    print(tabulate(rows, headers="keys", tablefmt="psql", floatfmt=".3f"))

@case_mark
def case_engine_card():
    """Generate a consolidated Markdown "engine card"."""
    plan = _load()
    lines = []

    def h(title):
        lines.append(f"\n## {title}\n")

    lines.append(f"# Engine Card: {plan.name}\n")

    h("Summary")
    for k, v in summary_dict(plan).items():
        lines.append(f"- **{k}**: {str(v).strip()}")

    h("Latency by layer type")
    lat = group_sum(plan.records, "type", "latency.avg_time")
    cnt = group_count(plan.records, "type")
    rows = [{"type": t, "count": cnt[t], "latency (ms)": lat[t]} for t in sorted(lat, key=lat.get, reverse=True)]
    lines.append(tabulate(rows, headers="keys", tablefmt="github", floatfmt=".4f"))

    h(f"Top {n_top_tactics} tactics")
    tactics = summarize_engine_tactics(plan, group_tactics=True, sort_key="latency")[:n_top_tactics]
    lines.append(tabulate(tactics, headers="keys", tablefmt="github", floatfmt=".3f"))

    h("Precision (bytes)")
    stats = compute_precision_stats(plan)
    prows = []
    for category, data in stats.items():
        for precision, size in data.items():
            prows.append({"category": category, "precision": precision, "bytes": int(size)})
    lines.append(tabulate(prows, headers="keys", tablefmt="github"))

    h("Lint hazards")
    reports = lint_engine(plan)
    total = sum(len(v) for v in reports.values())
    lines.append(f"Found {total} potential hazard(s).\n")
    for category, hazards in reports.items():
        for hz in hazards:
            lines.append(f"- **[{category}] {hz['hazard']}** - {hz['name']}")

    card_file.write_text("\n".join(lines))
    print(f"Wrote {card_file.name} ({len(lines)} lines)")

if __name__ == "__main__":
    case_summarize_tactics()
    case_engine_card()

    print("Finish")
