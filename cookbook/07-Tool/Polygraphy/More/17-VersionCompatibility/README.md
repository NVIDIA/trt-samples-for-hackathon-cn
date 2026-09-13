# Version compatibility and the lean runtime

+ What `version_compatible`, `exclude_lean_runtime` and `LoadRuntime` cost.

+ Steps to run.

```bash
python3 main.py
```

`CreateConfig(version_compatible=True)` produces a plan a *newer* TensorRT can
deserialize. What cannot be shown here is that promise being kept — it needs two
TensorRT versions and this image has one (11.1.0.106). Everything below is the
mechanism and the price, measured.

## The flag staples the lean runtime into the plan

Not a metaphor — the plan grows by the size of `libnvinfer_lean.so`:

```
baseline plan            :   13.190 MB
version_compatible plan  :  118.036 MB  (8.95x)
difference               :  104.847 MB
libnvinfer_lean.so.11.1.0:  104.845 MB  -- the plan carries a copy of it
the two agree to within 1528 bytes
a one-layer engine       :    0.011 MB ->  104.856 MB  (9221x)
```

The surcharge is a flat ~105 MB, not a percentage. A one-layer engine pays the
same 105 MB a real model does, so the smaller the model the worse the ratio
gets — 9221x for a single elementwise add.

## Getting the size back

```
baseline                       :   13.189 MB
version_compatible             :  118.035 MB
+ exclude_lean_runtime         :   13.189 MB  (+0 B vs baseline)
exclude_lean_runtime alone -> PolygraphyException: Cannot set EXCLUDE_LEAN_RUNTIME if version compatibility is not enabled.
loaded the excluded plan with LoadRuntime(libnvinfer_lean.so.11.1.0): I/O ['x', 'y', 'z']
```

`exclude_lean_runtime=True` returns the plan to *exactly* baseline size, and you
then owe the runtime at load time. That is the whole job of `LoadRuntime(path)`:
it creates a bootstrap `trt.Runtime` and calls `load_runtime` on it. The trade is
one shared 105 MB library against 105 MB inside each engine, so it wins as soon
as you ship more than one plan.

Setting `exclude_lean_runtime` without `version_compatible` is caught by
Polygraphy itself, with a sentence that says what to do — worth contrasting with
[`../13-PerLayerPrecision/`](../13-PerLayerPrecision/README.md), where the
removed loaders fail from two layers down inside pybind.

## The cost lands at load, not at inference

```
baseline           :   13.19 MB, deserialize    3.71 ms, infer 0.578 ms
version_compatible :  118.03 MB, deserialize  127.70 ms, infer 0.572 ms
+ exclude_lean     :   13.19 MB, deserialize    3.22 ms, infer 0.575 ms
```

**34.4x** on deserialization, and 6 µs on inference — noise. Once the engine
exists the kernels are the same kernels. So the flag hurts process startup,
container image size and anything that loads engines on demand, and does not
touch throughput.

## Deserializing it by hand returns `None`

```
raw trt.Runtime, host code not allowed, baseline           : engine
raw trt.Runtime, host code not allowed, version_compatible : None
raw trt.Runtime, host code not allowed, + exclude_lean     : engine
same plan after engine_host_code_allowed = True            : engine
```

Not an exception — `deserialize_cuda_engine` hands back `None` and logs

```
Cannot deserialize engine with lean runtime since IRuntime::getEngineHostCodeAllowed() is false
```

so code that does not check gets an `AttributeError` on the *next* line. An
embedded lean runtime is host code, and `getEngineHostCodeAllowed()` defaults to
false. Polygraphy's `EngineFromBytes` sets `runtime.engine_host_code_allowed =
True` for you inside a bare `try/except AttributeError`, so this only bites when
you deserialize with raw TensorRT.

Note the inversion against the names: the plan that **includes** the lean runtime
is the one a plain runtime refuses, and the plan that **excludes** it loads
fine — because excluding it means falling back to the linked `libnvinfer`, which
here is the same version that built the plan.

## The other compatibility axis

```
baseline             :  13.189 MB, infer 0.575 ms
AMPERE_PLUS          :  13.647 MB, infer 0.595 ms
SAME_COMPUTE_CAPABILITY:  13.190 MB, infer 0.597 ms
```

`version_compatible` is about the TensorRT version; `hardware_compatibility_level`
is about the GPU, and `AMPERE_PLUS` gives up architecture-specific kernels so one
plan runs on Ampere and later. On MNIST that is 0.46 MB and nothing outside run
to run variation — an honest result rather than a reassuring one. This model's
layers do not use the kernels the restriction takes away; nothing here shows that
a transformer would get off as lightly.

The two flags compose: version plus hardware compatibility is one plan for many
TensorRT versions and many GPUs.

## One piece of stale advice

Enabling either flag makes Polygraphy log:

> If you are using an ONNX model, please set the NATIVE_INSTANCENORM ONNX parser
> flag, e.g. `--onnx-flags NATIVE_INSTANCENORM`

On TensorRT 11 that made no difference in testing — building an
`InstanceNormalization` model version-compatible with and without
`trt.OnnxParserFlag.NATIVE_INSTANCENORM` produced plans of identical size
(104860708 bytes both times). Native InstanceNorm is the parser's default now.
