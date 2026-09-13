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
"""Data-parallel inference with Torch-TensorRT: one engine per GPU, the batch split across them.

Upstream `examples/distributed_inference/` covers two different things under one heading, and
only one of them runs here:

+ **Data parallel** -- the whole model on every GPU, the batch split. Scales throughput,
  needs no communication during inference, and is what this file measures.
+ **Tensor parallel** -- one model *split across* GPUs, with NCCL collectives inside the
  graph. **Blocked in this container**: it requires the TRT-LLM plugin library, and importing
  `torch_tensorrt` here reports

      CUDA 13 is not currently supported for TRT-LLM plugins.
      Please install pytorch with CUDA 12.x support

  The TensorRT-level equivalent is not blocked and is covered by
  `08-Advance/ContextParallelism`, which does the same collectives through `IDistCollective`
  without going near TRT-LLM.

The thing to get right in data parallel is that **each GPU needs its own compilation**. An
engine is device-specific (see `08-Advance/MultiDevice`), so a module compiled on device 0
cannot simply be moved; each rank compiles its own, and the batch is sharded across them.
"""

import threading
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch_tensorrt

from tensorrt_cookbook import case_mark

np.random.seed(31193)
torch.manual_seed(31193)

# Sized so one GPU's share takes milliseconds. At microsecond scale the Python barrier
# that releases the workers costs more than the inference, and the measurement becomes
# one of threading rather than of TensorRT -- see 08-Advance/MultiTask for that effect
# measured directly.
N_BATCH_PER_GPU = 256
N_FEATURE = 4096
N_LAYER = 16
N_WARMUP = 5
N_INFERENCE = 20

result = OrderedDict()

class Model(nn.Module):
    """Something with enough work per sample that scaling is visible."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_list = nn.ModuleList([nn.Linear(N_FEATURE, N_FEATURE) for _ in range(N_LAYER)])

    def forward(self, x):
        for layer in self.layer_list:
            x = torch.relu(layer(x))
        return x

def compile_on(device: int, state_dict) -> tuple:
    """Compile the model for one device. Returns `(module, example_input)`."""
    torch.cuda.set_device(device)
    model = Model().half().eval().to(f"cuda:{device}")
    model.load_state_dict({k: v.to(f"cuda:{device}") for k, v in state_dict.items()})
    example = torch.randn(N_BATCH_PER_GPU, N_FEATURE, dtype=torch.half, device=f"cuda:{device}")
    exported = torch.export.export(model, (example, ))
    compiled = torch_tensorrt.dynamo.compile(exported, inputs=[example], use_explicit_typing=True, min_block_size=1)
    return compiled, example

def measure(module, input_tensor, device: int) -> float:
    torch.cuda.set_device(device)
    for _ in range(N_WARMUP):
        module(input_tensor)
    torch.cuda.synchronize(device)
    latency_list = []
    for _ in range(N_INFERENCE):
        t0 = time.time()
        module(input_tensor)
        torch.cuda.synchronize(device)
        latency_list.append((time.time() - t0) * 1000)
    return float(np.median(latency_list))

# ================================================================ Cases

@case_mark
def case_compile_per_device() -> None:
    """Compile one engine per visible GPU, from identical weights."""
    device_count = torch.cuda.device_count()
    print(f"    visible GPUs: {device_count}")
    if device_count < 2:
        print("    Skip since no enough GPU is ready (need 2)")
        result["skip"] = True
        return

    torch.cuda.set_device(0)
    reference_model = Model().half().eval().cuda()
    state_dict = {k: v.cpu() for k, v in reference_model.state_dict().items()}

    module_list, input_list = [], []
    for device in range(device_count):
        t0 = time.time()
        compiled, example = compile_on(device, state_dict)
        module_list.append(compiled)
        input_list.append(example)
        print(f"    GPU {device}: compiled in {time.time() - t0:5.1f} s")
    result["device_count"] = device_count
    result["module_list"] = module_list
    result["input_list"] = input_list
    result["skip"] = False
    return

@case_mark
def case_same_answer_everywhere() -> None:
    """All replicas must agree, or the throughput number is meaningless."""
    if result.get("skip"):
        return
    reference = result["module_list"][0](result["input_list"][0]).float().cpu()
    same_input = result["input_list"][0]
    for device in range(1, result["device_count"]):
        moved = same_input.to(f"cuda:{device}")
        output = result["module_list"][device](moved).float().cpu()
        difference = float((output - reference).abs().max())
        print(f"    GPU {device} vs GPU 0 on identical input: max |diff| = {difference:.3e}")
        assert difference < 1e-2, f"Replica {device} disagrees with replica 0"
    return

@case_mark
def case_throughput_scaling() -> None:
    """One GPU against all of them, each holding its own shard of the batch."""
    if result.get("skip"):
        return
    device_count = result["device_count"]

    # Same shape of measurement as the parallel case: N_INFERENCE back-to-back, one sync
    module0, input0 = result["module_list"][0], result["input_list"][0]
    torch.cuda.set_device(0)
    for _ in range(N_WARMUP):
        module0(input0)
    torch.cuda.synchronize(0)
    t0 = time.time()
    for _ in range(N_INFERENCE):
        module0(input0)
    torch.cuda.synchronize(0)
    single = (time.time() - t0) * 1000 / N_INFERENCE
    print(f"    1 GPU : {single:7.3f} ms per round for {N_BATCH_PER_GPU} samples "
          f"= {N_BATCH_PER_GPU / single * 1000:9.0f} samples/s")
    result["single"] = single

    # Warm every replica before timing them together
    for device in range(device_count):
        measure(result["module_list"][device], result["input_list"][device], device)

    # Throughput, not per-round latency: each worker runs all its iterations back to back and
    # the wall time covers the whole run. Synchronising the workers every iteration would put
    # two Python barriers on the critical path per round, which at these speeds costs more
    # than the inference does -- that mistake is measured directly in 08-Advance/MultiTask.
    start_barrier = threading.Barrier(device_count + 1)
    done_barrier = threading.Barrier(device_count + 1)

    def worker(device: int) -> None:
        torch.cuda.set_device(device)  # per-thread, as always
        module, example = result["module_list"][device], result["input_list"][device]
        start_barrier.wait()
        for _ in range(N_INFERENCE):
            module(example)
        torch.cuda.synchronize(device)
        done_barrier.wait()

    thread_list = [threading.Thread(target=worker, args=(device, ), daemon=True) for device in range(device_count)]
    for thread in thread_list:
        thread.start()

    start_barrier.wait()
    t0 = time.time()
    done_barrier.wait()
    total_ms = (time.time() - t0) * 1000
    for thread in thread_list:
        thread.join(timeout=30)

    parallel = total_ms / N_INFERENCE
    total_batch = N_BATCH_PER_GPU * device_count
    print(f"    {device_count} GPUs: {parallel:7.3f} ms per round for {total_batch} samples "
          f"= {total_batch / parallel * 1000:9.0f} samples/s")
    result["parallel"] = parallel
    return

@case_mark
def case_summary() -> None:
    if result.get("skip"):
        print("    Skipped: fewer than 2 GPUs")
        return
    device_count = result["device_count"]
    single_throughput = N_BATCH_PER_GPU / result["single"] * 1000
    parallel_throughput = N_BATCH_PER_GPU * device_count / result["parallel"] * 1000
    print("\n" + "    " + "=" * 66)
    print(f"    {'configuration':<22}{'batch':>8}{'ms':>10}{'samples/s':>14}{'speed-up':>12}")
    print("    " + "-" * 66)
    print(f"    {'1 GPU':<22}{N_BATCH_PER_GPU:>8}{result['single']:>10.3f}{single_throughput:>14.0f}{1.0:>11.2f}x")
    print(f"    {f'{device_count} GPUs, data parallel':<22}{N_BATCH_PER_GPU * device_count:>8}{result['parallel']:>10.3f}{parallel_throughput:>14.0f}{parallel_throughput / single_throughput:>11.2f}x")
    print("    " + "=" * 66)
    print("    Data parallel needs no communication during inference, so the ceiling is the number")
    print("    of GPUs; what eats into it is the host-side cost of driving them (see 08-Advance/MultiTask).")
    return

def main() -> None:
    case_compile_per_device()
    case_same_answer_everywhere()
    case_throughput_scaling()
    case_summary()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
