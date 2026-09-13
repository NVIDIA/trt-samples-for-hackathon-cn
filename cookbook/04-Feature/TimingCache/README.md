# Timing Cache

+ Usage of timing cache to reduce engine building time, including editable timing cache.

+ Steps to run.

```bash
python3 main.py
```

## Cases

| Case | What it shows |
| ---- | ------------- |
| 0-8 | Building with / without a timing cache, and reusing one cache across two similar networks. Watch the reported build times and the growth of `model.TimingCache`. |
| 9 `case_editable` | The editable-timing-cache **API**: `BuilderFlag.EDITABLE_TIMING_CACHE`, `queryKeys()`, `query()`, `update()`, then serialize and rebuild from the edited cache. |
| 10 `case_force_tactic` | The editable-timing-cache **use case**: override which tactic a layer uses. |

## Forcing a tactic (`case_force_tactic`)

A layer usually has several implementations ("tactics"). TensorRT profiles them all and keeps the
fastest. Sometimes another one is wanted — to reproduce a known-good build, to avoid a tactic that
misbehaves on particular inputs, or simply to make builds deterministic. Writing the desired tactic
into the timing cache before the build reads it does exactly that.

The catch is discovering *which other tactics exist*: *there is no API for it*. TensorRT only lists
them in the **verbose build log**:

```txt
Autotuning op node_linear(key: 0xe5c222cd6b9e29cbe66c8bec07cb6a07): [ONNX Layer: node_linear]
Sorted table of all evaluated tactics:
tactic_id, cost(in ms), cost/fastest_cost, prediction_correlation, kernel_name, tactic_hash, tunable_parameter
   4, 0.0126784, 1.00000, 0.84378, sm80_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize32x32x64_..., 0x2c80ce2f5c4a6,
  34, 0.0130784, 1.03155, 0.92579, sm86_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x64x64_..., 0x650de5b566787,
  ...
The selected tactic is (tactic hash, cost(in ms)):0x2c80ce2f5c4a6, 0.0126784
```

So the flow is: build once with a verbose logger, parse that table, pick a tactic, write it into the
cache with `update()`, and build again.

```txt
Autotuned operators with more than one tactic: 2 / 2
Operator node_linear: TensorRT picked 0x2c80ce2f5c4a6, forcing 0x650de5b566787 (cost 0.0131040 ms) instead
Cache update accepted: True
Tactic in the cache after the build: 0x650de5b566787
Forced tactic survived the build: True
```

The last two lines are the verification. Had TensorRT re-profiled the operator, it would have
written the *fastest* tactic back into the cache; finding the forced one still there means the build
consumed the edited entry instead of measuring again.

### Why the first build runs in a child process

The tactic table only exists in the process output, so it has to be captured. Redirecting file
descriptors 1 and 2 in-process while TensorRT and CUDA hold them open **segfaults the runtime**, and
subclassing `trt.ILogger` in Python to record the messages does too. Running that one build as a
child process (`main.py --profile-build`) and reading its `stdout` / `stderr` is what works, so
`profile_build_in_subprocess()` does that; the child leaves the cache on disk for the parent to edit.

### Caveats

+ Tactic hashes are specific to the GPU architecture, the TensorRT version and the layer
  configuration. A hash harvested on one machine is meaningless on another.
+ Forcing a slower tactic is, by construction, slower. This is a determinism / debugging tool, not
  an optimization.
+ Which operators appear in the table depends on the network. Layers that get fused into a single
  Myelin node expose no per-layer tactic choice at all, so a network built purely of fusable
  operators produces an empty table — `case_force_tactic` reports this and returns instead of
  failing.
