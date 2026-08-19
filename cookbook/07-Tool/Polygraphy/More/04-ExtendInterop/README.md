# Interoperating with TensorRT (`func.extend`)

+ Reach through Polygraphy to the raw TensorRT API without leaving the lazy style.

+ Steps to run.

```bash
python3 main.py
```

## The mechanism

```python
@func.extend(NetworkFromOnnxPath(onnx_file))
def load_and_rename(builder, network, parser):
    network.name = "CookbookMnist"      # raw TensorRT API
    # no return needed -- `extend` handles it

build_engine = EngineFromNetwork(load_and_rename, config=CreateConfig())
```

The decorated function receives whatever the wrapped loader produced and returns
nothing. `EngineFromNetwork` is handed the *function*, not a call to it, so the
chain stays lazy — this is the only way to edit a network in the lazy style,
because with lazy loaders there is no network yet
([`../01-LazyVsImmediate/`](../01-LazyVsImmediate/README.md)).

It works the same on a config loader:

```
raw   config.set_flag(BuilderFlag.REFIT) -> engine refittable = True
typed CreateConfig(refittable=True)      -> engine refittable = True
```

`extend` is a hatch, not a different mechanism.

## The hatch stops where TensorRT stops

The upstream example for this feature sets `trt.BuilderFlag.FP16` directly,
presented as the way around Polygraphy not supporting a flag. **On TensorRT 11
that line does not run at all:**

```
trt.BuilderFlag.FP16 : REMOVED in TensorRT 11
trt.BuilderFlag.INT8 : REMOVED in TensorRT 11
trt.BuilderFlag.BF16 : REMOVED in TensorRT 11
trt.BuilderFlag.TF32 : present
AttributeError: type object 'tensorrt.tensorrt.BuilderFlag' has no attribute 'FP16'
```

So `CreateConfig(fp16=True)` raising (see
[`../02-ComparingBackends/`](../02-ComparingBackends/README.md)) is **not**
Polygraphy being conservative — it is reporting an API that no longer exists.
There is nothing to reach around. Precision is now declared on the network, not
requested from the builder; see the cookbook skill `trt-strong-typing-migration`.

What is left: `DEBUG`, `DIRECT_IO`, `DISABLE_COMPILATION_CACHE`,
`DISABLE_TIMING_CACHE`, `DISTRIBUTIVE_INDEPENDENCE`, `EDITABLE_TIMING_CACHE`,
`ERROR_ON_TIMING_CACHE_MISS`, `EXCLUDE_LEAN_RUNTIME`, `GPU_FALLBACK`,
`MONITOR_MEMORY`, `REFIT`, `REFIT_IDENTICAL`, `REFIT_INDIVIDUAL`, `SAFETY_SCOPE`,
`SPARSE_WEIGHTS`, `STRICT_NANS`, `STRIP_PLAN`, `TF32`, `VERSION_COMPATIBLE`,
`WEIGHT_STREAMING`.

## The trap: a runner reuses its output buffers

```
the two inputs really do give different results : True
kept reference now matches the SECOND result    : True   <- first result lost
deepcopy still differs from the second result   : True   <- survived
```

Keep a reference to `outputs["y"]`, call `infer()` again, and the reference you
kept now shows the **new** result. The old values are gone, nothing was raised,
and the variable name still says what you meant.

`outputs` is a view into the runner, not a snapshot. Anything that has to outlive
the next `infer()` needs `copy.deepcopy`. Upstream mentions this in a comment;
this case shows what it costs.

## Related

+ [`../01-LazyVsImmediate/`](../01-LazyVsImmediate/README.md) — why editing a network needs `extend` at all.
+ [`../02-ComparingBackends/`](../02-ComparingBackends/README.md) — the `CreateConfig(fp16=True)` refusal this case explains.
