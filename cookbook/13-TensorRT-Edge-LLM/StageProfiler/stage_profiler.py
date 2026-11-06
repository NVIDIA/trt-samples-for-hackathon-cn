# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
"""An `IProfiler` that groups layer times by a caller-declared stage.

Adapted from `cpp/profiling/layerProfiler.{h,cpp}` of
https://github.com/NVIDIA/TensorRT-Edge-LLM (Apache-2.0), where the stages are an LLM's prefill and
decode. Nothing is copied: theirs is a C++ singleton holding
`LayerProfile{name, stageName, timeMs[], iterationCount}`, this is the same data model in Python.

`CookbookProfiler` in `tensorrt_cookbook` reports one flat stream of `(layer, time)` pairs. That is
the right thing when an engine is run one way. It stops being the right thing as soon as the same
engine is run in two regimes -- two optimization profiles, two input shapes, a warm-up pass and a
measured pass -- because the two sets of times then land in one undifferentiated list and the
averages describe a workload that never ran.
"""

from __future__ import annotations

from collections import OrderedDict

import tensorrt as trt

class StageProfiler(trt.IProfiler):
    """Collect per-layer times, tagged with whichever stage is current.

    ```python
    profiler = StageProfiler()
    context.profiler = profiler
    with profiler.stage("prefill"):
        context.execute_async_v3(stream)
    with profiler.stage("decode"):
        context.execute_async_v3(stream)
    print(profiler.format_report())
    ```

    Upstream makes this a singleton (`LayerProfiler::getInstance()`, copy constructor and
    assignment deleted) because a C++ runtime has several execution contexts that must report into
    one table. Here an ordinary object is enough and is easier to reason about: TensorRT calls
    `report_layer_time` on whatever object the context's `profiler` attribute points at.
    """

    #: Times recorded while no stage is open land here rather than being dropped.
    DEFAULT_STAGE = "(unlabelled)"

    def __init__(self) -> None:
        super().__init__()
        # stage -> layer -> [time_ms, ...]. Ordered so the report follows execution order, which is
        # what makes two stages comparable line by line.
        self.record: OrderedDict[str, OrderedDict[str, list[float]]] = OrderedDict()
        self.n_iteration: dict[str, int] = {}
        self._current_stage = self.DEFAULT_STAGE

    # -- the IProfiler callback ------------------------------------------------------------------

    def report_layer_time(self, layer_name: str, time_ms: float) -> None:
        """Called by TensorRT once per layer per `execute_*`. Must stay cheap."""
        stage = self.record.setdefault(self._current_stage, OrderedDict())
        stage.setdefault(layer_name, []).append(time_ms)

    # -- stage control ---------------------------------------------------------------------------

    def set_stage(self, name: str) -> None:
        self._current_stage = name
        self.n_iteration[name] = self.n_iteration.get(name, 0)

    def stage(self, name: str):
        """Context manager form, so a stage cannot be left open by an exception."""

        class _Stage:

            def __init__(self, profiler: StageProfiler, stage_name: str):
                self.profiler = profiler
                self.stage_name = stage_name
                self.previous = profiler._current_stage

            def __enter__(self):
                self.profiler.set_stage(self.stage_name)
                self.profiler.n_iteration[self.stage_name] = self.profiler.n_iteration.get(self.stage_name, 0) + 1
                return self.profiler

            def __exit__(self, *_):
                self.profiler._current_stage = self.previous
                return False

        return _Stage(self, name)

    # -- reporting -------------------------------------------------------------------------------

    def stage_total_ms(self, stage: str) -> float:
        """Sum of every layer's mean time in one stage."""
        layer_map = self.record.get(stage, {})
        return sum(sum(v) / len(v) for v in layer_map.values() if v)

    def summary(self) -> dict:
        return {
            stage: {
                "n_layer": len(layer_map),
                "n_iteration": self.n_iteration.get(stage, 0),
                "total_ms": self.stage_total_ms(stage),
            }
            for stage, layer_map in self.record.items()
        }

    def format_report(self, *, n_top: int = 5) -> str:
        """One block per stage, slowest layers first."""
        line_list = []
        for stage, layer_map in self.record.items():
            total = self.stage_total_ms(stage)
            line_list.append(f"  stage {stage!r}: {len(layer_map)} layers, "
                             f"{self.n_iteration.get(stage, 0)} iteration(s), {total * 1000:.1f} us total")
            ranked = sorted(((sum(v) / len(v), k) for k, v in layer_map.items() if v), reverse=True)
            for mean_ms, name in ranked[:n_top]:
                share = mean_ms / total * 100 if total > 0 else 0.0
                line_list.append(f"      {mean_ms * 1000:8.2f} us  {share:5.1f}%  {name[:70]}")
        return "\n".join(line_list)

    def format_comparison(self, stage_a: str, stage_b: str, *, n_top: int = 8) -> str:
        """Per-layer A vs B, for the question the flat profiler cannot answer.

        Layers present in only one stage are shown rather than skipped: on a real engine that is
        usually the finding, not a nuisance.
        """
        map_a = self.record.get(stage_a, {})
        map_b = self.record.get(stage_b, {})

        def mean(layer_map, name):
            value_list = layer_map.get(name)
            return sum(value_list) / len(value_list) if value_list else None

        name_list = list(map_a) + [n for n in map_b if n not in map_a]
        row_list = []
        for name in name_list:
            a, b = mean(map_a, name), mean(map_b, name)
            delta = (b - a) if (a is not None and b is not None) else None
            row_list.append((abs(delta) if delta is not None else float("inf"), name, a, b))
        row_list.sort(reverse=True)

        line_list = [f"  {'layer':<46}{stage_a:>12}{stage_b:>12}{'delta':>12}"]
        for _, name, a, b in row_list[:n_top]:
            sa = f"{a * 1000:.2f}us" if a is not None else "absent"
            sb = f"{b * 1000:.2f}us" if b is not None else "absent"
            sd = f"{(b - a) * 1000:+.2f}us" if (a is not None and b is not None) else "-"
            line_list.append(f"  {name[:45]:<46}{sa:>12}{sb:>12}{sd:>12}")
        return "\n".join(line_list)
