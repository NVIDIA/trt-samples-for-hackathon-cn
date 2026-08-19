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

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import onnx
import tensorrt as trt
import torch
import torch.nn as nn
import yaml

from tensorrt_cookbook import case_mark

HERE = Path(__file__).parent
MODEL_A = HERE / "model-a.onnx"
MODEL_B = HERE / "model-b.onnx"
SHAPE = (8, 3, 128, 128)

class Net(nn.Module):
    """A conv stack. Q/DQ placement only matters when there are fusions to help or break."""

    def __init__(self, width=(3, 32, 64, 64, 128, 128), head=16):
        super().__init__()
        self.block = nn.ModuleList([nn.Sequential(nn.Conv2d(width[i], width[i + 1], 3, padding=1), nn.BatchNorm2d(width[i + 1]), nn.ReLU()) for i in range(len(width) - 1)])
        self.head = nn.Conv2d(width[-1], head, 1)

    def forward(self, x):
        for i, block in enumerate(self.block):
            x = block(x)
            if i % 2 == 1:
                x = torch.max_pool2d(x, 2)
        return self.head(x)

def export(path: Path, seed: int, head: int) -> None:
    if path.exists():
        return
    torch.manual_seed(seed)
    model = Net(head=head).eval()
    torch.onnx.export(model, (torch.randn(SHAPE), ), str(path), input_names=["x"], output_names=["y"], dynamo=False)
    return

def run_autotune(model: Path, output_dir: Path, *, schemes: int, pattern_cache: Path = None) -> dict:
    """Drive the CLI and pull the numbers back out of its log and state file."""
    shutil.rmtree(output_dir, ignore_errors=True)
    command = [
        sys.executable,
        "-m",
        "modelopt.onnx.quantization.autotune",
        "--onnx_path",
        str(model),
        "--output_dir",
        str(output_dir),
        "--mode",
        "quick",
        "-s",
        str(schemes),
        "--use_trtexec",  # see case_backend
    ]
    if pattern_cache is not None:
        command += ["--pattern_cache", str(pattern_cache)]
    t0 = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, cwd=HERE)
    elapsed = time.perf_counter() - t0
    log = process.stdout + process.stderr
    if process.returncode != 0:
        raise RuntimeError(f"autotune failed rc={process.returncode}:\n{log[-1500:]}")

    result = {"elapsed": elapsed, "log": log}
    match = re.search(r"Results: ([\d.]+) ms → ([\d.]+) ms \(([\d.]+)x speedup\)", log)
    if match:
        result["baseline_ms"] = float(match.group(1))
        result["tuned_ms"] = float(match.group(2))
        result["speedup"] = float(match.group(3))
    result["n_benchmark"] = len(re.findall(r"TrtExec benchmark \(median\)", log))
    state = output_dir / "autotuner_state.yaml"
    if state.exists():
        parsed = yaml.safe_load(state.read_text())
        result["baseline_latency_ms"] = parsed.get("baseline_latency_ms")
        result["patterns"] = [p["pattern_signature"] for p in parsed.get("patterns", [])]
        result["n_scheme"] = sum(len(p.get("schemes", [])) for p in parsed.get("patterns", []))
    return result

def count_qdq(path: Path) -> tuple:
    model = onnx.load(str(path))
    q = sum(node.op_type == "QuantizeLinear" for node in model.graph.node)
    dq = sum(node.op_type == "DequantizeLinear" for node in model.graph.node)
    return q, dq, len(model.graph.node)

@case_mark
def case_backend():
    """Which benchmark backend works here, and why the default one does not.

    The autotuner can time candidates either through the TensorRT Python API
    (`TensorRTPyBenchmark`, the default) or by shelling out to `trtexec`
    (`TrtExecBenchmark`, `--use_trtexec`). On TensorRT 11 only the second one runs.
    """
    print(f"    tensorrt {trt.__version__}")
    print(f"    NetworkDefinitionCreationFlag members: {[x for x in dir(trt.NetworkDefinitionCreationFlag) if x.isupper()]}")
    has_explicit_batch = hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH")
    print(f"    NetworkDefinitionCreationFlag.EXPLICIT_BATCH exists: {has_explicit_batch}")
    assert not has_explicit_batch, "EXPLICIT_BATCH is back; re-check whether --use_trtexec is still needed"
    print("\n    `modelopt/onnx/quantization/autotune/benchmark.py:361` does")
    print("        self.network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)")
    print("    which raises AttributeError on TensorRT 10+, where explicit batch is the only mode")
    print("    and the flag was removed. The autotuner catches it and reports only")
    print("        ERROR - Failed to initialize TensorRT benchmark")
    print("    without naming the cause. **Every run in this example therefore passes")
    print("    `--use_trtexec`**, which takes a different code path and works. This is the same")
    print("    family of breakage as the `plugin_creator_list` shim in 05-Plugin/ONNXPTQWithPlugin.")
    return

@case_mark
def case_autotune():
    """What the search actually buys, timed by TensorRT rather than guessed."""
    export(MODEL_A, seed=31193, head=16)
    print(f"    model-a.onnx: {count_qdq(MODEL_A)[2]} nodes, no Q/DQ yet")

    result = run_autotune(MODEL_A, HERE / "out-a", schemes=4)
    print(f"    autotuned in {result['elapsed']:.1f} s, {result['n_benchmark']} TensorRT benchmarks")
    print(f"    baseline {result['baseline_ms']:.3f} ms -> tuned {result['tuned_ms']:.3f} ms"
          f"  (**{result['speedup']:.3f}x**)")

    q, dq, n = count_qdq(HERE / "out-a" / "optimized_final.onnx")
    qb, dqb, nb = count_qdq(HERE / "out-a" / "baseline.onnx")
    print(f"    baseline.onnx        : {nb:>3} nodes, {qb} Q / {dqb} DQ")
    print(f"    optimized_final.onnx : {n:>3} nodes, {q} Q / {dq} DQ")
    assert result["speedup"] >= 1.0, result["speedup"]
    print("\n    Note what is being optimized. Both graphs quantize the same tensors; what moves is")
    print("    **where the Q/DQ pairs sit relative to the fusions**, and the only way to know which")
    print("    placement wins is to build the engine and time it. That is the whole idea: the")
    print("    objective function is a real TensorRT latency, not a proxy like 'quantize more ops'.")
    print("    Contrast 07-Tool/FP16Tuning, which searches precision per layer against *accuracy*.")
    return

@case_mark
def case_regions_and_patterns():
    """The unit of search is a region, and regions are keyed by their op signature."""
    state = yaml.safe_load((HERE / "out-a" / "autotuner_state.yaml").read_text())
    print(f"    baseline_latency_ms: {state['baseline_latency_ms']}")
    print(f"    performance_threshold: {state['config']['performance_threshold']}"
          "   <- a scheme must beat the best by this factor to be adopted")
    print(f"\n    {len(state['patterns'])} pattern(s) discovered, each a sub-graph the search treats as one unit:")
    for pattern in state["patterns"]:
        signature = pattern["pattern_signature"]
        # The signature carries the attributes, so two Convs with different kernels are
        # different patterns and cannot share a cached scheme.
        short = " -> ".join(op.split("[")[0] for op in signature.split("->"))
        best = min((s["latency_ms"] for s in pattern["schemes"]), default=float("nan"))
        print(f"      size {pattern['pattern_size']}: {short:<45} {len(pattern['schemes'])} scheme(s), best {best:.4f} ms")
    print("\n    A pattern signature is the op sequence *with its attributes*:")
    print(f"      {state['patterns'][0]['pattern_signature'][:110]}...")
    print("    so `Conv[kernel_shape=3x3]` and `Conv[kernel_shape=1x1]` are different patterns and")
    print("    never share a result. That is what makes the cache below safe to reuse across models.")
    return

@case_mark
def case_pattern_cache():
    """Reuse what was learned on one model when tuning a similar one."""
    export(MODEL_B, seed=97, head=24)  # same blocks, different head
    cache = HERE / "out-a" / "autotuner_state_pattern_cache.yaml"
    cached = yaml.safe_load(cache.read_text())
    print(f"    cache from model-a: {len(cached['pattern_schemes'])} pattern(s), "
          f"{sum(len(p['schemes']) for p in cached['pattern_schemes'])} scheme(s), "
          f"minimum_distance={cached['minimum_distance']}")

    cold = run_autotune(MODEL_B, HERE / "out-b-cold", schemes=4)
    warm = run_autotune(MODEL_B, HERE / "out-b-warm", schemes=4, pattern_cache=cache)
    print("\n    tuning model-b        benchmarks   wall time   speedup found")
    print("    " + "-" * 62)
    for label, r in [("without the cache", cold), ("with model-a's cache", warm)]:
        print(f"    {label:<21} {r['n_benchmark']:>10}   {r['elapsed']:>7.1f} s   {r.get('speedup', float('nan')):.3f}x")

    print("\n    The cache is keyed by pattern signature, so the shared conv blocks are recognised")
    print("    even though model-b has a different head. What it saves is benchmarks -- and a")
    print("    benchmark is a full engine build plus a timing run, which is the expensive part.")
    if warm["n_benchmark"] < cold["n_benchmark"]:
        print(f"    Here it saved {cold['n_benchmark'] - warm['n_benchmark']} of {cold['n_benchmark']}.")
    else:
        print(f"    **Here it saved nothing**: {warm['n_benchmark']} benchmarks either way. Both models are")
        print("    small enough that the search explores every scheme regardless. The mechanism is")
        print("    real; the benefit needs a model with more repeated structure than this one.")
    if abs(cold.get("speedup", 0) - warm.get("speedup", 0)) > 0.05:
        print(f"\n    Note also that the two runs found **different optima** ({cold['speedup']:.3f}x without")
        print(f"    the cache, {warm['speedup']:.3f}x with it) on the *same* model. The search is stochastic")
        print("    -- it mutates the top schemes and stops at a scheme budget -- so a single run is a")
        print("    sample, not the answer. Do not read a cache-vs-no-cache latency difference as an")
        print("    effect of the cache; only the benchmark count is a fair comparison here.")
    return

def main() -> None:
    case_backend()
    case_autotune()
    case_regions_and_patterns()
    case_pattern_cache()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
