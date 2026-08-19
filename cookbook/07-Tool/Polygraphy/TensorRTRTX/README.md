# TensorRTRTX

+ Driving **TensorRT-RTX** from Polygraphy instead of TensorRT, and the four places where the two
  backends are not interchangeable.

+ Everything else in `07-Tool/Polygraphy/` runs against TensorRT 11. One environment variable
  redirects the whole tool — CLI and Python API — at `tensorrt_rtx` instead, and the interesting
  part is what does *not* survive the switch.

## Running

```bash
./main.sh
```

Needs `tensorrt_rtx` (`pip install tensorrt_rtx`); the script prints a skip notice and exits 0 if it
is missing. Measured on **1 x B200, polygraphy 0.50.3, TensorRT-RTX 1.6.1.120, TensorRT 11.1.0.106**
— the two TensorRT packages are separate Python modules and coexist in one environment.

## 1. The switch, and when it is read

```bash
POLYGRAPHY_USE_TENSORRT_RTX=0 polygraphy run model.onnx --trt -vv   # Loaded Module: tensorrt     11.1.0.106
POLYGRAPHY_USE_TENSORRT_RTX=1 polygraphy run model.onnx --trt -vv   # Loaded Module: tensorrt_rtx  1.6.1.120
```

`polygraphy/config.py` evaluates the variable **once, at import time**:

```python
USE_TENSORRT_RTX = bool(os.environ.get("POLYGRAPHY_USE_TENSORRT_RTX", "0") != "0")
```

So in a Python program it has to be set **before `import polygraphy`**. `rtx_api.py` sets it
afterwards on purpose and reports `config.USE_TENSORRT_RTX = False` — no warning, no error, the whole
run silently uses the wrong backend. Requires polygraphy >= 0.49.24.

## 2. Precision flags: rejected by the API, silently dropped by the CLI

TensorRT-RTX has no weak-typing precision flags, and polygraphy knows it
(`backend/trt/config.py:525`). The two front ends disagree about what to do with that:

| Route | Result |
| ----- | ------ |
| `CreateConfig(fp16=True)` | `PolygraphyException: Precision flags (fp16, int8, bf16, fp8) are not supported with USE_TENSORRT_RTX=1.` — same for `int8`, `bf16`, `fp8` |
| `polygraphy run --trt --fp16` | **`PASSED`.** The verbose log shows the engine was built with `Flags | [TF32, STRONGLY_TYPED]` |

The CLI neither honours the flag nor complains about it. A script that passes `--fp16` and checks
only the exit code will report success for an engine that was never built in FP16. Use the Python
API, or grep the verbose log for the flag list, if it matters.

## 3. The same model is numerically looser under RTX

Same ONNX, same comparison harness, only the backend changed:

| Backend | `max_absdiff` | `max_reldiff` | tol 1e-5 | tol 1e-4 |
| ------- | ------------- | ------------- | -------- | -------- |
| `tensorrt` 11.1.0.106 | 7.4506e-09 | 1.8437e-05 | **100%** | 100% |
| `tensorrt_rtx` 1.6.1.120 | 1.5311e-05 | 0.11365 | **0%** | 100% |

RTX is ~2000x further from ONNX Runtime on the same FP32 network, and fails polygraphy's default
tolerance. It is not a wrong answer — the `int64` argmax output `z` matches **exactly** on both, and
the `rel=0.11` comes from one near-zero element of `y` — the failure mode `check_array`'s
`b_relative` and `07-Tool/Polygraphy/DebugWorkflow` are both about. **Do not carry a tolerance tuned against TensorRT
over to TensorRT-RTX unchanged**, and do not read the resulting failure as a broken engine before
looking at which elements moved.

## 4. `--compute-capabilities`: only on `convert`, and it validates the name

Ahead-of-time targeting is exposed by **`polygraphy convert` only** — `tools/convert/convert.py:62`
is the single place that passes `allow_compute_capabilities=True`:

```
run     --compute-capabilities 8.9  -> [E] Unrecognized Options: ['--compute-capabilities', '8.9']
convert --compute-capabilities 10.0 -> [!] Compute capability 10.0 (SM100) not supported by this TensorRT-RTX version.
convert --compute-capabilities 8.9  -> 13,906,484 bytes
convert --compute-capabilities 12.0 -> 13,907,172 bytes   (a different plan)
```

**This is the strongest reason to drive RTX through polygraphy.** Asking `tensorrt_rtx` directly for
an unsupported target gives you nothing: `set_compute_capability` returns `False` and the build
quietly produces a plan for the current device (see
[`10-TensorRT-RTX/08-DeployTimeRuntimeConfig`](../../../10-TensorRT-RTX/08-DeployTimeRuntimeConfig/README.md)).
Polygraphy checks the enumerator by name first and says so. It also does the slot allocation for
you — `config.num_compute_capabilities = len(...)` before `set_compute_capability`, which is the step
that is easy to miss by hand.

`--use-gpu` is the `ComputeCapability.CURRENT` shorthand, mutually exclusive with
`--compute-capabilities`. Both are rejected outright when `USE_TENSORRT_RTX` is not set.

## 5. Concurrent builds share one timing cache safely

`--save-timing-cache` works under RTX and is guarded by `polygraphy.util.LockFile`
(`util/util.py:360`), which takes an exclusive lock on a sibling `<path>.lock`. Four builds writing
one cache concurrently:

```
4 builds finished, failures: 0
shared.cache 172 bytes, lock file present: yes
```

Note the lock is on the **timing** cache. TensorRT-RTX's other cache, `IRuntimeCache`, is a
deploy-side artefact polygraphy does not manage — that one is
[`10-TensorRT-RTX/08-DeployTimeRuntimeConfig`](../../../10-TensorRT-RTX/08-DeployTimeRuntimeConfig/README.md).
