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
"""Reusing built TensorRT engines across compilations, sessions and weight changes.

Building an engine is the expensive part of Torch-TensorRT. AOT
(`torch_tensorrt.dynamo.compile`) pays it once per process; JIT (`torch.compile`)
pays it again every time a graph is invalidated. Engine caching writes each built
engine to disk keyed by a hash of the PyTorch sub-graph, so a later compilation
can load it instead of rebuilding.

Two switches, both needed for the cache to be useful:

    cache_built_engines=True   write engines into the cache
    reuse_cached_engines=True  read them back

and one precondition: `immutable_weights=False`. A cached engine is reused by
**refitting** it with the current weights, and only a refittable engine can do
that.

The model here is `resnet18` with **random** weights on purpose -- the cache hash
covers the graph, not the values, so pretrained weights would only add a download.
`case_weight_agnostic` proves that and checks the refit is numerically right.
"""

import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch_tensorrt as torch_trt
import torchvision.models as models
from torch_tensorrt.dynamo._defaults import TIMING_CACHE_PATH
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache

from tensorrt_cookbook import case_mark

torch.manual_seed(31193)

output_path = Path(__file__).parent
cache_path = output_path / "engine_cache"
SHAPE = (8, 3, 224, 224)
N_ITERATION = 3

def build_model() -> torch.nn.Module:
    """`resnet18`, randomly initialised. Weights are irrelevant to the cache."""
    return models.resnet18(weights=None).cuda().eval()

def reset_cache() -> None:
    """Start from an empty engine cache and an empty timing cache.

    The timing cache is a *different* mechanism (it remembers kernel timings, not
    engines) and would otherwise speed up the uncached baseline and blur the
    comparison. Clearing it is only needed to measure; leave it alone in practice.
    """
    shutil.rmtree(cache_path, ignore_errors=True)
    Path(TIMING_CACHE_PATH).unlink(missing_ok=True)
    return

def n_cached_engine() -> int:
    """How many engines are sitting in the cache directory."""
    return len(list(cache_path.iterdir())) if cache_path.is_dir() else 0

def compile_aot(model: torch.nn.Module, *, b_cache: bool, engine_cache=None) -> tuple:
    """One AOT compilation, returning `(compiled module, seconds)`."""
    Path(TIMING_CACHE_PATH).unlink(missing_ok=True)
    torch._dynamo.reset()
    example = (torch.randn(SHAPE).cuda(), )
    exported = torch.export.export(model, example)

    option = dict(min_block_size=1, immutable_weights=False, cache_built_engines=b_cache, reuse_cached_engines=b_cache)
    if engine_cache is None:
        option.update(engine_cache_dir=str(cache_path), engine_cache_size=1 << 30)
    else:
        option.update(custom_engine_cache=engine_cache)

    t0 = time.time()
    compiled = torch_trt.dynamo.compile(exported, example, **option)
    return compiled, time.time() - t0

def compile_jit(model: torch.nn.Module, *, b_cache: bool) -> float:
    """One JIT compilation through the `tensorrt` backend, returning seconds."""
    Path(TIMING_CACHE_PATH).unlink(missing_ok=True)
    torch._dynamo.reset()
    data = torch.rand(SHAPE).cuda()

    t0 = time.time()
    compiled = torch.compile(
        model,
        backend="tensorrt",
        options={
            "min_block_size": 1,
            "immutable_weights": False,
            "cache_built_engines": b_cache,
            "reuse_cached_engines": b_cache,
            "engine_cache_dir": str(cache_path),
        },
    )
    with torch.no_grad():
        compiled(data)  # `torch.compile` is lazy, this is what triggers the build
    return time.time() - t0

def report(title: str, second_list: list) -> None:
    """Print the three timings and the speed-up they imply."""
    print(f"    {title}")
    for tag, second in zip(["no cache        ", "build + save    ", "load from cache "], second_list):
        print(f"      {tag}: {second:7.2f} s")
    print(f"      -> reusing is {second_list[0] / second_list[2]:.0f}x faster than building")
    return

@case_mark
def case_aot_dynamo_compile() -> None:
    """AOT: `torch_tensorrt.dynamo.compile`, three compilations of the same model.

    1. caching off, the baseline
    2. caching on, cache empty -- builds *and* writes, so it is the slowest
    3. caching on, cache warm -- loads and refits
    """
    reset_cache()
    model = build_model()
    second_list = [compile_aot(model, b_cache=b_cache)[1] for b_cache in [False, True, True]]
    report("AOT (torch_tensorrt.dynamo.compile)", second_list)
    print(f"      engines in {cache_path.name}/: {n_cached_engine()}")
    return

@case_mark
def case_jit_torch_compile() -> None:
    """JIT: the same three steps through `torch.compile(backend="tensorrt")`.

    This is where caching matters most. AOT pays the build cost once per process,
    but a JIT graph is rebuilt whenever dynamo invalidates it, so without a cache
    the cost is paid over and over.
    """
    reset_cache()
    model = build_model()
    second_list = [compile_jit(model, b_cache=b_cache) for b_cache in [False, True, True]]
    report("JIT (torch.compile, backend=tensorrt)", second_list)
    print(f"      engines in {cache_path.name}/: {n_cached_engine()}")
    return

@case_mark
def case_weight_agnostic() -> None:
    """The cache key covers the graph, not the weights.

    Compile once, then replace **every** parameter and compile again. The cache
    still hits, and the engine is refitted with the new values rather than
    rebuilt. That is what makes caching useful for fine-tuned or LoRA-style
    checkpoints that share one architecture.

    The numeric check is the point: a cache hit that failed to refit would
    silently return the *previous* weights' answer.
    """
    reset_cache()
    model = build_model()
    compile_aot(model, b_cache=True)  # Populate
    n_before = n_cached_engine()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.02)
        data = torch.randn(SHAPE).cuda()
        reference = model(data)

    compiled, second = compile_aot(model, b_cache=True)
    with torch.no_grad():
        output = compiled(data)
    difference = (reference - output).abs().max().item()

    print(f"    every parameter replaced, then recompiled in {second:.2f} s")
    print(f"    engines in cache: {n_before} -> {n_cached_engine()} (unchanged, so the hash ignored the weights)")
    print(f"    max |eager - TensorRT| = {difference:.3e} -> the engine was refitted, not just reused")
    assert difference < 1e-2, "cache hit returned the old weights, refit did not happen"
    return

class RAMEngineCache(BaseEngineCache):
    """A cache that lives in a dict instead of on disk.

    Only `save` and `load` have to be implemented. Subclassing this way is how a
    team would put engines in shared or remote storage; the counters here are
    just so the case can show the hit.
    """

    def __init__(self) -> None:
        """Create an empty in-memory cache."""
        self.engine_cache: Dict[str, bytes] = {}
        self.n_save, self.n_hit, self.n_miss = 0, 0, 0

    def save(self, hash: str, blob: bytes) -> None:
        """Insert one engine blob."""
        self.engine_cache[hash] = blob
        self.n_save += 1
        return

    def load(self, hash: str) -> Optional[bytes]:
        """Return the blob for `hash`, or `None`."""
        if hash in self.engine_cache:
            self.n_hit += 1
            return self.engine_cache[hash]
        self.n_miss += 1
        return None

@case_mark
def case_custom_cache() -> None:
    """Swap the on-disk cache for a `BaseEngineCache` subclass."""
    reset_cache()
    model = build_model()
    engine_cache = RAMEngineCache()
    second_list = [compile_aot(model, b_cache=b_cache, engine_cache=engine_cache)[1] for b_cache in [False, True, True]]
    report("AOT with a custom in-memory cache", second_list)
    print(f"      RAMEngineCache: {len(engine_cache.engine_cache)} blob(s), "
          f"{engine_cache.n_save} save(s), {engine_cache.n_hit} hit(s), {engine_cache.n_miss} miss(es)")
    print(f"      nothing on disk: {n_cached_engine()} file(s) in {cache_path.name}/")
    return

if __name__ == "__main__":
    case_aot_dynamo_compile()
    case_jit_torch_compile()
    case_weight_agnostic()
    case_custom_cache()
    reset_cache()

    print("\nFinish")
