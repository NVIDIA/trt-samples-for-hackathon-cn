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
"""Sample GPU utilisation, memory and power beside a timed run, without being able to break it.

Adapted from `apps/benchmark/trtmc_benchmark/telemetry.py` in
https://github.com/NVIDIA/TensorRT-Model-Connect (Apache-2.0). Their design decision is the whole
point and is kept: telemetry is **best-effort**. A missing `nvidia-smi`, a parse failure or a
sampling timeout records an error string and the benchmark carries on. A measurement harness that
dies because its optional instrumentation is unavailable is worse than one with no instrumentation.

Three rules this follows, each of which is easy to get wrong:

1. The sampler runs on a daemon thread, so it can never hold the process open.
2. It samples *beside* the timed region, never inside it -- `nvidia-smi` is a subprocess launch of
   tens of milliseconds and would land in the numbers it is meant to describe.
3. It honours `CUDA_VISIBLE_DEVICES`, because the device your code calls "0" is not necessarily the
   device `nvidia-smi` calls "0".
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import threading
import time
from typing import Any

class GpuTelemetry:
    """Context manager that samples `nvidia-smi` on a background thread.

    ```python
    with GpuTelemetry(interval_ms=50) as telemetry:
        ...  # the work being measured
    print(telemetry.summary())
    ```

    Attributes after the block:
        samples: list of dicts, one per successful sample
        error: a string when sampling could not run or stopped, otherwise None
    """

    FIELD_LIST = ("index", "name", "gpu_utilization_percent", "memory_used_mib", "power_w")
    _QUERY = "index,name,utilization.gpu,memory.used,power.draw"

    def __init__(self, interval_ms: int = 50, mode: str = "auto") -> None:
        assert mode in ("auto", "off"), f"mode must be 'auto' or 'off', got {mode!r}"
        assert interval_ms > 0, "interval_ms must be positive"
        self.mode = mode
        self.interval_seconds = interval_ms / 1000.0
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def __enter__(self) -> GpuTelemetry:
        self._started_at = time.monotonic()
        if self.mode == "off":
            self.error = "telemetry disabled by mode='off'"
            return self
        if shutil.which("nvidia-smi") is None:
            # Recorded, not raised. This is the branch that keeps a container without the NVIDIA
            # tooling from failing a benchmark that otherwise runs fine.
            self.error = "nvidia-smi was not found on PATH"
            return self
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            # Bounded join: one sampling interval plus a second. A hung `nvidia-smi` must not
            # become a hung benchmark.
            self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))
            if self._thread.is_alive():
                self.error = "telemetry thread did not stop in time"

    def _device_selector(self) -> str | None:
        """The first entry of CUDA_VISIBLE_DEVICES, which is what the process calls device 0."""
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", maxsplit=1)[0].strip()
        return visible or None

    def _sample_loop(self) -> None:
        command = ["nvidia-smi", f"--query-gpu={self._QUERY}", "--format=csv,noheader,nounits"]
        selector = self._device_selector()
        if selector is not None:
            command += ["--id", selector]
        while not self._stop.is_set():
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=5.0, check=False)
                if completed.returncode == 0:
                    self._record(completed.stdout)
                elif self.error is None:
                    self.error = f"nvidia-smi exited {completed.returncode}: {completed.stderr.strip()[:120]}"
            except Exception as e:  # noqa: BLE001 - telemetry must not raise into the caller
                if self.error is None:
                    self.error = f"{type(e).__name__}: {e}"
            # `wait` rather than `sleep`, so __exit__ stops the loop immediately instead of after
            # the current interval has elapsed.
            self._stop.wait(self.interval_seconds)

    def _record(self, stdout: str) -> None:
        elapsed_ms = (time.monotonic() - self._started_at) * 1000.0
        for row in csv.reader(io.StringIO(stdout)):
            if len(row) != len(self.FIELD_LIST):
                continue
            sample: dict[str, Any] = {"elapsed_ms": round(elapsed_ms, 3)}
            for key, value in zip(self.FIELD_LIST, [cell.strip() for cell in row]):
                if key == "name":
                    sample[key] = value
                elif key == "index":
                    sample[key] = int(value) if value.isdigit() else value
                else:
                    try:
                        sample[key] = float(value)
                    except ValueError:  # "[N/A]" on a GPU that does not report that field
                        sample[key] = None
            self.samples.append(sample)

    def summary(self) -> dict[str, Any]:
        """Min / mean / max per numeric field, plus the sample count and any error."""
        result: dict[str, Any] = {"n_sample": len(self.samples), "error": self.error}
        if not self.samples:
            return result
        # One poll yields one row per visible GPU, so `n_sample` is rows, not polls. Without
        # CUDA_VISIBLE_DEVICES every GPU on the machine is sampled and the statistics below mix
        # them -- on a 2-GPU host an idle second card drags `memory_used_mib.min` to near zero.
        index_set = {s.get("index") for s in self.samples}
        result["n_device"] = len(index_set)
        result["n_poll"] = len(self.samples) // max(len(index_set), 1)
        result["device"] = self.samples[0].get("name")
        for field in ("gpu_utilization_percent", "memory_used_mib", "power_w"):
            value_list = [s[field] for s in self.samples if isinstance(s.get(field), float)]
            if value_list:
                result[field] = {
                    "min": min(value_list),
                    "mean": sum(value_list) / len(value_list),
                    "max": max(value_list),
                }
        return result

    def format_summary(self) -> str:
        summary = self.summary()
        line_list = [f"    rows={summary['n_sample']} ({summary.get('n_poll', 0)} polls x {summary.get('n_device', 0)} GPU), device={summary.get('device', 'unknown')}"]
        if summary["error"] is not None:
            line_list.append(f"    [WARN] {summary['error']}")
        for field in ("gpu_utilization_percent", "memory_used_mib", "power_w"):
            if field in summary:
                stat = summary[field]
                line_list.append(f"    {field:<26}min={stat['min']:9.2f}  mean={stat['mean']:9.2f}  max={stat['max']:9.2f}")
        return "\n".join(line_list)
