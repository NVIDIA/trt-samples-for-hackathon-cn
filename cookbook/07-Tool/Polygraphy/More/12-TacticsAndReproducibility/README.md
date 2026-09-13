# Tactics and build reproducibility

+ Pinning tactic choices across builds, now that Polygraphy's tactic replay is
  gone.

+ Steps to run.

```bash
python3 main.py
```

## The replay API does not exist any more

`TacticRecorder` / `TacticReplayer` recorded which tactic TensorRT picked per
layer and forced the same choice on a later build. Two separate deaths stacked
on top of each other — Polygraphy deprecated them, and the TensorRT base class
they subclass was removed:

```
TacticReplayData(): TacticReplayData -- constructs fine, it is just a dict
TacticRecorder: AttributeError: module 'tensorrt' has no attribute 'IAlgorithmSelector'
  warned first: TacticRecorder is deprecated and will be removed in Polygraphy 0.55.0.
TacticReplayer: AttributeError: module 'tensorrt' has no attribute 'IAlgorithmSelector'
trt.IAlgorithmSelector present in TensorRT 11.1.0.106: False
IBuilderConfig.algorithm_selector: False
```

`TacticReplayData` survives because it is only an `OrderedDict` and never touches
TensorRT — the same shape of trap as `Calibrator` in
[`../06-Int8IsNowExplicit/`](../06-Int8IsNowExplicit/README.md): the object
constructs, so the code reads as healthy until it meets the builder. This is the
fourth upstream Polygraphy example that does not run on TensorRT 11.

## `save_timing_cache=` is write-only

The timing cache took over the job, and Polygraphy's only kwarg for it saves the
cache **after** the build. Nothing feeds it back in. Passing the same path on
every build looks like caching and is not:

```
warm-up (CUDA init, discarded): 11.22 s
no cache at all               :  8.43 s
save_timing_cache=, run 1     :  8.36 s  (wrote 32825 B)
save_timing_cache=, run 2     :  8.38 s  <- the file is there and full, and nothing got faster
speed-up from run 1 to run 2  : 1.00x
```

The file grows, the contents are correct, and every tactic is re-measured anyway.
The warm-up build matters here: the first build in a process pays for CUDA
initialisation, so without discarding it, build #1 looks slow and build #2 looks
like the cache worked.

`CreateConfig` has no kwarg for loading one, so it goes on by hand — which is
[`../04-ExtendInterop/`](../04-ExtendInterop/README.md) applied to a real need:

```python
@func.extend(CreateConfig())
def config_with_cache_loaded(builder, network, config):
    config.set_timing_cache(config.create_timing_cache(cache_file.read_bytes()), ignore_mismatch=False)
```

```
no cache                      :  8.38 s
cache loaded onto the config  :  7.05 s
speed-up                      : 1.19x  -- against 1.00x for the kwarg alone
```

## How you would have noticed

`ERROR_ON_TIMING_CACHE_MISS` turns a silent miss into a build failure — any layer
TensorRT still has to measure aborts the build:

```
flag set, no cache loaded  : PolygraphyException -- build refused
flag set, cache loaded     : built
```

Worth setting in CI on a build that is supposed to be fully cached. Not worth
leaving on in normal use, since any model change legitimately misses.

## What the cache holds

`queryKeys()` / `query()` expose one entry per autotuned layer, holding the
winning tactic hash and its measured time — the same information the removed
replay API recorded, reached through TensorRT instead of Polygraphy:

```
entries in timing.cache: 30
  tactic 0x4b9ef35ad8721, measured 0.006938 ms
  tactic 0x0, measured 0.004906 ms
```

That covers *reusing* the tactic that was measured. Forcing a **different** one —
to reproduce a known-good build or dodge a misbehaving tactic — needs
`EDITABLE_TIMING_CACHE` and is covered in
[`04-Feature/TimingCache/`](../../../../04-Feature/TimingCache/README.md).
