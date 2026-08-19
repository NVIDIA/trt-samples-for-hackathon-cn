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
"""Where does context parallelism start to pay off?

Attention costs O(sequence^2) while the collectives cost O(sequence), so splitting
the sequence across GPUs is a loss on short inputs and a win on long ones. This
script sweeps the sequence length through `main.py` and prints where the crossover
actually is on this machine.

Run `python3 main.py` first if you only want the single default configuration.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

N_SEQUENCE_LIST = [2048, 4096, 8192, 16384, 32768, 65536]

output_path = Path(__file__).parent
result_file = output_path / "data-result.json"

def main() -> None:
    row_list = []
    for n_sequence in N_SEQUENCE_LIST:
        environment = os.environ.copy()
        environment["TRT_N_SEQUENCE"] = str(n_sequence)
        process = subprocess.run([sys.executable, str(output_path / "main.py")], env=environment, cwd=str(output_path), capture_output=True, text=True)
        if process.returncode != 0:
            print(f"[sequence={n_sequence}] FAILED\n{process.stdout[-2000:]}")
            continue
        result = json.loads(result_file.read_text())
        if "context_parallel" not in result:  # Fewer than 2 GPUs, `main.py` skipped itself
            print(process.stdout.strip())
            return
        assert result["n_sequence"] == n_sequence, f"Stale {result_file.name}: it reports sequence {result['n_sequence']}, expected {n_sequence}"
        row_list.append((n_sequence, result["single_device"], result["context_parallel"]))
        print(f"[sequence={n_sequence:>6}] single={result['single_device']['latency_ms']:7.3f} ms, parallel={result['context_parallel']['latency_ms']:7.3f} ms")

    print("\n" + "=" * 100)
    print(f"{'Sequence':>10}{'SingleGPU(ms)':>16}{'2xGPU(ms)':>14}{'Speedup':>10}{'RelDiff':>12}{'SM-clock(MHz)':>16}{'Verdict':>18}")
    print("-" * 100)
    for n_sequence, single, parallel in row_list:
        speedup = single["latency_ms"] / parallel["latency_ms"]
        verdict = "context parallel" if speedup > 1.0 else "single GPU"
        clock = f"{single['sm_clock']}/{parallel['sm_clock']}"
        print(f"{n_sequence:>10}{single['latency_ms']:>16.3f}{parallel['latency_ms']:>14.3f}{speedup:>10.2f}{parallel['relative']:>12.2e}{clock:>16}{verdict:>18}")
    print("=" * 100)
    # If a row was measured on a throttled GPU its clock collapses; say so instead of
    # quietly publishing it. See the thermal-guard note in `main.py`.
    slow = [n for n, single, parallel in row_list if min(single["sm_clock"], parallel["sm_clock"]) < 1500]
    if slow:
        print(f"WARNING: sequence {slow} were measured below 1500 MHz (thermal throttling), do not trust those rows.")

    winning = [n for n, single, parallel in row_list if single["latency_ms"] / parallel["latency_ms"] > 1.0]
    if winning:
        print(f"Crossover: context parallelism wins from sequence {min(winning)} upwards.")
    else:
        print("Context parallelism never wins in the swept range, the collectives dominate.")
    print("Below the crossover the two NCCL AllGathers on K/V cost more than the attention")
    print("work they save, so a 2-GPU run is slower than just doing the whole thing on one.")
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
