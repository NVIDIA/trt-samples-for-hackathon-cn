# AliasedIOPlugin

+ A Python plugin that writes into one of its own inputs, using the aliased-I/O capability of `IPluginV3OneBuildV2`.

+ Before `IPluginV3OneBuildV2`, plugin inputs were strictly read-only, so an in-place operator had
  to copy the whole input to the output first even when it only touched a few elements.
  `IPluginV3OneBuildV2` adds `get_aliased_input(output_index)`, which declares that an output shares
  its buffer with an input.

+ The operator is scatter-add — `data[index[i]] += updates[i]` — the aggregation step of a graph
  neural network, where each node sums the features of its neighbours. `data` is an accumulator, so
  adding into it directly is the whole point; `atomicAdd` is required because several neighbours may
  target the same node.

+ Steps to run.

```bash
python3 main.py
```

## Relation to the other in-place examples

| Example                        | Language | Aliased I/O                                    | Operator                     |
| ------------------------------ | -------- | ----------------------------------------------- | ---------------------------- |
| `05-Plugin/InPlacePlugin`      | C++      | yes — `v_2_0::IPluginV3OneBuild` + `getAliasedInput` | elementwise `AddScalar`  |
| `05-Plugin/AliasedIOPlugin`    | Python   | yes — `IPluginV3OneBuildV2` + `get_aliased_input`    | scatter-add              |
| `02-API/Layer/KVCacheUpdate`   | Python   | aliasing produced by a *layer*, not a plugin     | KV-cache update              |

`InPlacePlugin` already shows the same capability in C++, but its `AddScalar` would work just as
well without aliasing. Scatter-add is an operator that genuinely needs to accumulate into its own
input, which is what makes the aliasing worth declaring.

## Three things that are easy to get wrong

+ **The preview feature must be enabled at build time.** Without
  `builder_config.set_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03, True)` the build
  fails with:

```txt
Error Code 4: Internal Error (Aliased I/O used by plugin (Unnamed Layer* 0) [PluginV3_V3ONE] but
PreviewFeature::kALIASED_PLUGIN_IO_10_03 not enabled.)
```

+ **`ICudaEngine.get_aliased_input_tensor()` does not report plugin aliasing.** It returns the
  aliased input for aliasing introduced by *layers* such as KVCacheUpdate; for a plugin it returns
  `None` even though the plugin declared `get_aliased_input(0) == 0`. `main.py` prints this so the
  behaviour is visible rather than assumed. The caller therefore has to honour the contract itself:

```python
tw.context.set_tensor_address("data_out", tw.buffer["data"][1])  # Point the output at its aliased input
```

+ **The result lands in the input's buffer.** After inference the output's own host buffer is still
  whatever it was (all zeros in the log), because the plugin wrote into `data`'s device memory. Read
  the answer back from there — `main.py` does an explicit `cudaMemcpy` from `tw.buffer["data"][1]`.
  `05-Plugin/InPlacePlugin` shows the same effect from the C++ side.
