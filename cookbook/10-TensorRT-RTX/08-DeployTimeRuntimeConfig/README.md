# 08-DeployTimeRuntimeConfig

+ What TensorRT-RTX lets you change **after** the engine is built, measured rather than enumerated.

> **Optional package.** This example needs `tensorrt_rtx`, which is **not** part of the base
> environment. Install it with `pip install tensorrt_rtx` (it pulls `tensorrt_rtx_cu13{,_libs,_bindings}`
> and coexists with `tensorrt` — they are separate Python modules and neither shadows the other).
> Without it every case here prints `[SKIP] tensorrt_rtx is not installed` and exits 0.

+ `01`–`07` each poke one RTX-only API. This example is about the deploy side as a workflow: the
  plan is a fixed artefact and `IRuntimeConfig` is most of what is left. Four knobs are measured
  here, and the RTX features that this machine cannot run are measured too, so the boundary is on
  the page instead of in someone's head.

## Running

```bash
python3 main.py
```

Requires `tensorrt_rtx` (`pip install tensorrt_rtx`). Measured on **1 x B200, TensorRT-RTX
1.6.1.120**, alongside the system TensorRT 11.1.0.106 — the two packages coexist, they are separate
Python modules.

## What it measures

### 1. The runtime cache, across process boundaries

`IRuntimeCache` holds JIT-compiled kernels. Serializing it in-process and reading it straight back
(what `03-RuntimeCache` does) proves the API works but cannot show what it is for, because the JIT
result is already in memory. The cost only appears in a **fresh process**, so every row below is one.

There are **two** caches stacked, and only one of them ships with your application:

| Cache | Where | Yours? |
| ----- | ----- | ------ |
| CUDA driver PTX→SASS cache | `~/.nv/ComputeCache`, shared by every process on the machine | no — disable with `CUDA_CACHE_DISABLE=1` |
| TensorRT-RTX `IRuntimeCache` | wherever you serialize it | yes |

Measured, first inference in a fresh process:

| Driver cache | cold (no cache file) | warm (cache file) | runtime cache worth |
| ------------ | -------------------- | ----------------- | ------------------- |
| enabled      | 158.9 ms             | 22.4 ms           | 7.1x                |
| `CUDA_CACHE_DISABLE=1` | 650.1 ms   | 36.8 ms           | **17.7x**           |

The cache blob is 535,936 bytes and the output checksum is unchanged — a runtime cache never moves
the numerics.

**The lesson is the second row, and the reason is the first.** With the driver cache left enabled
the cold number is whatever that shared on-disk cache happens to hold: repeated runs here gave 155,
163, 413 and 654 ms, moving the apparent benefit between ~7x and ~28x without anything about
TensorRT changing. The warm number is stable at 22–37 ms either way. **Pin the driver cache before
quoting a JIT-warmup speedup, or you are reporting the state of `~/.nv` as if it were a property of
your engine.**

For deployment: a server that restarts without shipping this file pays the compile again, and
cannot rely on the driver cache to hide it.

### 2. Dynamic-shape kernel specialization — `EAGER` is the slow one

Each row is its own process with an empty runtime cache. `n=8` is the profile's `opt` shape, `n=13`
is a shape the engine has never seen.

| Strategy | `n=8` first | `n=13` first (median of 3) | `n=13` again |
| -------- | ----------- | -------------------------- | ------------ |
| `NONE`   | 151–173 ms  | **12.7 ms**                | ~7.9 ms      |
| `LAZY`   | 156–174 ms  | **12.9 ms**                | ~8.0 ms      |
| `EAGER`  | 193–232 ms  | **35.4 ms**                | ~8.2 ms      |

`EAGER` is **2.8x slower** than `NONE` at an unseen shape — the opposite of what the name suggests.
Eager specialization compiles a shape-specialized kernel up front; on this engine that costs more
than the generic kernel it replaces saves. Steady state is identical for all three, and all three
agree numerically (two distinct checksums, one per shape, across all nine runs).

### 3. CUDA graph strategy — no effect here, and that is the point

| Strategy | Steady state (200 iterations) |
| -------- | ----------------------------- |
| `DISABLED` | 0.2033 ms |
| `WHOLE_GRAPH_CAPTURE` | 0.2025 ms |

Ratio 1.00x. Six convolutions at 8x64x64x64 are compute bound, so there is no launch overhead left
for graph capture to remove. Reach for it when the engine is many small kernels, not a few large
ones — the same conclusion `08-Advance/MultiTask` reached for TensorRT proper.

Both measurements were taken at 1965 MHz / 26 C; the script prints the SM clock beside every latency
because a throttled clock silently invalidates all of them.

### 4. Ahead-of-time compute-capability targeting, and how to get it wrong

`num_compute_capabilities` is a **writable property**, not a read-only count of what the build
supports. It starts at 0, and the slots have to be allocated before they can be written:

```python
builder_config.set_compute_capability(trt.ComputeCapability.SM89, 0)   # -> False, no slots
builder_config.num_compute_capabilities = 2                            # allocate them
builder_config.set_compute_capability(trt.ComputeCapability.SM89, 0)   # -> True
```

Done properly, the target really is baked in — five distinct plans for the same network:

| Target | `set_compute_capability` | plan `sha256[:16]` |
| ------ | ------------------------ | ------------------ |
| unset    | —     | `8767df856172486a` |
| `CURRENT`| True  | `ecbb02a1f607040b` |
| `SM75`   | True  | `d528fefa763d3aa9` |
| `SM89`   | True  | `c6102ef88f2253c8` |
| `SM120`  | True  | `f87d275685bb4a6c` |

**The failure mode is the lesson.** At the default 0 slots `set_compute_capability` returns `False`
rather than raising, and the build then silently produces a plan for the *current* device instead of
the one you asked for. Check the return value — a cross-compiled plan that quietly targets the build
machine is indistinguishable from a correct one until it reaches the deploy machine.

## What does not work on this machine

Measured by `case_unavailable_here`, not assumed:

| Feature | Result |
| ------- | ------ |
| `STRIP_PLAN` + `REFIT` weightless engine | **cannot build.** `REFIT` alone fails too, on a bare 4x4 MatMul, with `Myelin ... CUDA error 222 loading a module`. Fails for every target compute capability, including `SM89` and `SM120` |
| `get_engine_validity()` preflight | answers `INVALID` (reason code 2) for **every** plan, including ones it then deserializes and runs. A false negative, so it cannot be used as a preflight |

`trt.ComputeCapability` enumerates SM75, SM80, SM86, SM89, SM120, SM121. This is a **B200, compute
capability 10.0** — SM100 is not in that set, and TensorRT-RTX is off-label on a datacenter part. The
weightless build-here / refit-and-run-there flow needs an actual RTX GPU. Re-run this example there
before believing anything in this section is permanent.

## Gotcha

`IRuntimeConfig` does **not** keep the runtime cache alive. Dropping the last Python reference after
`set_runtime_cache(cache)` **segfaults the process**, with no error from TensorRT. Hold it:

```python
cache = runtime_config.create_runtime_cache()   # must stay in scope
runtime_config.set_runtime_cache(cache)
```
