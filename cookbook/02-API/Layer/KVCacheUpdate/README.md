# KV cache update layer

+ KV cache update layer.

+ Steps to run.

```bash
python3 main.py
```

+ Write newly computed key/value vectors into an existing KV cache at per-batch positions. The update is performed in place: the output shares its device memory address with the `cache` input. Requires a strongly-typed network. Refer to `case_simple`.

+ Available values of `trt.KVCacheMode`.

| Name   | Comment                                                                                                                                                     |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LINEAR | For each batch element `i` and sequence position `s`: `output[i, :, writeIndices[i] + s, :] = update[i, :, s, :]`, i.e. contiguous write starting at `writeIndices[i]`. |

+ Input / output tensors.

| Tensor       | Role     | Data Type                              | Shape             | Notes                                             |
| ------------ | -------- | -------------------------------------- | ----------------- | ------------------------------------------------- |
| cache        | Input    | `float32`, `float16`, `bfloat16` (`T`) | `[b, d, s_max, h]`| Must be a network input with static `s_max`       |
| update       | Input    | `T`                                    | `[b, d, s, h]`    | `s <= s_max`                                       |
| writeIndices | Input    | `int32`, `int64` (`M`)                 | `[b]`             | `writeIndices[i] + s <= s_max`                    |
| output       | Output   | `T`                                    | `[b, d, s_max, h]`| In-place update; aliases the `cache` input memory |

+ Shape: `b` = batch size, `d` = number of heads, `s_max` = max (static) sequence length, `s` = update sequence length, `h` = head size.

+ Notes.

+ Because the output aliases the `cache` input, the example queries `engine.get_aliased_input_tensor(...)` and binds the output buffer to the aliased input address before inference.

+ DLA is not supported.

## Attributes

| Attribute | Description                                                                                                                                                     | Default |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| cacheMode | Specifies the cache update mode (currently only `LINEAR`). See the `trt.KVCacheMode` table above.                                                             | LINEAR  |
