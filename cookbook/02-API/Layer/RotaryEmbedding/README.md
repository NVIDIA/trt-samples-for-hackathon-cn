# RotaryEmbedding layer

+ RotaryEmbedding layer.

+ Steps to run.

```bash
python3 main.py
```

+ Apply Rotary Position Embedding (RoPE) to the input tensor using cosine/sine caches and optional position ids, see `case_simple`.

+ Input / output data type and shape:
  + Input tensor of type `T` with shape `[b, d, s, h]`; `cos_cache` and `sin_cache` of type `T` with shape `[b, s, h/2]` or `[maxPositionId+1, h/2]`; optional `position_ids` of type `M` with shape `[b, s]`.
  + Output tensor of type `T` with the same shape `[b, d, s, h]`.
  + `T` in [float16, float32, bfloat16], `M` (position_ids) is int64.
  + When `rotary_embedding_dim != 0`, the last dimension of the caches becomes `rotary_embedding_dim/2` instead of `h/2`.

## Attributes

| Attribute | Description | Default | Range / Notes |
| --- | --- | --- | --- |
| `interleaved` | Whether the input tensor is in interleaved format, i.e., whether the 2-d vectors rotated are taken from adjacent 2 elements in the hidden dimension. | `False` | Boolean |
| `rotary_embedding_dim` | The hidden dimension participating in RoPE computation. A special value of 0 means the full hidden dimension participates in RoPE. | `0` | Integer >= 0 |
