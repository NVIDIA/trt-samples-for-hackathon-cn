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
"""Pinning tactic choices across builds, now that tactic replay is gone.

Polygraphy's `TacticRecorder` / `TacticReplayer` recorded the tactic TensorRT
picked for each layer and forced the same choice on a later build. Both are
deprecated in Polygraphy 0.50.3, and the TensorRT API they were built on --
`trt.IAlgorithmSelector` and the whole `IAlgorithm*` family -- **no longer exists
in TensorRT 11**. The upstream example for this feature does not run.

The timing cache took over the job. This example covers the Polygraphy side of
that, and the measurement that matters:

    EngineFromNetwork(..., save_timing_cache=path)   writes the cache
                                                     and never reads it back

Passing the same path on every build looks like caching and is not. The number
is in `case_save_timing_cache_does_not_load_it`.

Forcing one specific tactic (rather than reusing whatever was measured) is the
editable timing cache, covered in `04-Feature/TimingCache/`.
"""

import time
from pathlib import Path

import tensorrt as trt
from polygraphy import func
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

onnx_file = str(cookbook_path("00-Data", "model", "model-trained.onnx"))
cache_file = Path(__file__).parent / "timing.cache"

def build(config, save_timing_cache=None) -> float:
    """Build the engine once, return the wall clock in seconds."""
    t0 = time.perf_counter()
    EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=config, save_timing_cache=save_timing_cache)()
    return time.perf_counter() - t0

@func.extend(CreateConfig())
def config_with_cache_loaded(builder, network, config) -> None:
    """`CreateConfig` has no kwarg for loading a timing cache, so do it by hand.

    This is `../04-ExtendInterop/` applied to a real need: reach the native
    `IBuilderConfig` while keeping the loader chain lazy.
    """
    config.set_timing_cache(config.create_timing_cache(cache_file.read_bytes()), ignore_mismatch=False)

@case_mark
def case_the_replay_api_is_gone() -> None:
    """`TacticRecorder` / `TacticReplayer` cannot be constructed on TensorRT 11.

    Two separate deaths stacked on top of each other: Polygraphy deprecated them
    (removal in 0.55.0), and the `trt.IAlgorithmSelector` base class they subclass
    was removed from TensorRT. The failure is an `AttributeError` from inside
    Polygraphy, at construction.

    `TacticReplayData` still works, because it is only an `OrderedDict` of layer
    name -> algorithm and never touches TensorRT. Code that builds one looks
    healthy right up to the point it is handed to a recorder.
    """
    import warnings

    from polygraphy.backend.trt import TacticRecorder, TacticReplayData, TacticReplayer

    print(f"    TacticReplayData(): {type(TacticReplayData()).__name__} -- constructs fine, it is just a dict")

    for name, loader, argument in [("TacticRecorder", TacticRecorder, "replay.json"), ("TacticReplayer", TacticReplayer, TacticReplayData())]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                loader(argument)
                print(f"    {name}: unexpectedly constructed")
            except Exception as e:
                print(f"    {name:<14}: {type(e).__name__}: {e}")
            for warning in caught:
                print(f"      warned first: {warning.message}")

    print(f"    trt.IAlgorithmSelector present in TensorRT {trt.__version__}: {hasattr(trt, 'IAlgorithmSelector')}")
    print(f"    IBuilderConfig.algorithm_selector: {hasattr(trt.IBuilderConfig, 'algorithm_selector')}")
    return

@case_mark
def case_save_timing_cache_does_not_load_it() -> None:
    """The trap: `save_timing_cache=` is write-only.

    Polygraphy's only timing-cache kwarg saves the cache after the build (and
    merges it with whatever was already at that path). Nothing in that path
    feeds the cache *into* the next build, so passing it on every build gets the
    file written and the tactics re-measured every time.

    The first build in a process pays for CUDA initialisation, so a throwaway
    warm-up runs first -- otherwise build #1 looks slow and build #2 looks like
    the cache worked.
    """
    print(f"    warm-up (CUDA init, discarded): {build(CreateConfig()):5.2f} s")
    cache_file.unlink(missing_ok=True)

    baseline = build(CreateConfig())
    first = build(CreateConfig(), save_timing_cache=str(cache_file))
    second = build(CreateConfig(), save_timing_cache=str(cache_file))
    print(f"    no cache at all               : {baseline:5.2f} s")
    print(f"    save_timing_cache=, run 1     : {first:5.2f} s  (wrote {cache_file.stat().st_size} B)")
    print(f"    save_timing_cache=, run 2     : {second:5.2f} s  <- the file is there and full, and nothing got faster")
    print(f"    speed-up from run 1 to run 2  : {first / second:.2f}x")
    return

@case_mark
def case_loading_it_properly() -> None:
    """Load the cache onto the config by hand, and the saving becomes worth it.

    Same file, same model, same machine -- the only change is that the build now
    reads the cache it was writing all along.
    """
    baseline = build(CreateConfig())
    loaded = build(config_with_cache_loaded)
    print(f"    no cache                      : {baseline:5.2f} s")
    print(f"    cache loaded onto the config  : {loaded:5.2f} s")
    print(f"    speed-up                      : {baseline / loaded:.2f}x  -- against 1.00x for the kwarg alone")
    return

@case_mark
def case_proving_the_cache_was_used() -> None:
    """`ERROR_ON_TIMING_CACHE_MISS` turns a silent miss into a build failure.

    This is the answer to "how would I have noticed". With the flag set, any
    layer TensorRT has to measure aborts the build, so a cache that is not
    actually being consumed cannot pass unnoticed -- which is exactly what the
    `save_timing_cache=` kwarg was doing.

    Worth using in CI on a build that is supposed to be fully cached; not worth
    leaving on in normal use, since any model change legitimately misses.
    """

    def attempt(config, tag: str) -> None:
        try:
            EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=config)()
            print(f"    {tag}: built")
        except Exception as e:
            print(f"    {tag}: {type(e).__name__} -- build refused")

    @func.extend(CreateConfig())
    def strict_without_cache(builder, network, config):
        config.set_flag(trt.BuilderFlag.ERROR_ON_TIMING_CACHE_MISS)

    @func.extend(CreateConfig())
    def strict_with_cache(builder, network, config):
        config.set_flag(trt.BuilderFlag.ERROR_ON_TIMING_CACHE_MISS)
        config.set_timing_cache(config.create_timing_cache(cache_file.read_bytes()), ignore_mismatch=False)

    attempt(strict_without_cache, "flag set, no cache loaded  ")
    attempt(strict_with_cache, "flag set, cache loaded     ")
    print("    the same flag would have caught the kwarg-only setup on day one")
    return

@case_mark
def case_what_the_cache_actually_holds() -> None:
    """The cache is the tactic record that `TacticReplayData` used to be.

    `queryKeys()` / `query()` expose one entry per autotuned layer, each holding
    the winning tactic hash and its measured time -- the same information the
    removed replay API recorded, reached through TensorRT instead of Polygraphy.

    Overwriting an entry to force a *different* tactic needs
    `EDITABLE_TIMING_CACHE`; `04-Feature/TimingCache/` does that.
    """
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    config = builder.create_builder_config()
    cache = config.create_timing_cache(cache_file.read_bytes())
    key_list = cache.queryKeys()
    print(f"    entries in {cache_file.name}: {len(key_list)}")
    for key in key_list[:3]:
        value = cache.query(key)
        print(f"      tactic {hex(value.tacticHash)}, measured {value.timingMSec:.6f} ms")
    print("    one entry per autotuned layer -- this is where the tactic choice now lives")
    return

if __name__ == "__main__":
    case_the_replay_api_is_gone()
    case_save_timing_cache_does_not_load_it()
    case_loading_it_properly()
    case_proving_the_cache_was_used()
    case_what_the_cache_actually_holds()

    print("\nFinish")
