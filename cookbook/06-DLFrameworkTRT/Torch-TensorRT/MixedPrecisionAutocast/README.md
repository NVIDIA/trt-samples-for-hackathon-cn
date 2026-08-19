# Mixed Precision with Autocast

+ Getting FP16 / BF16 out of a **strongly typed** TensorRT 11 network.

+ Steps to run.

```shell
python3 main.py
```

## The trap first: the old way is silently a no-op

```
enabled_precisions={torch.float16}    :  0.151 ms, out float32   max rel err 0.00e+00
enable_autocast=True  (the fix)       :  0.130 ms, out float16   max rel err 1.76e-03
```

On a weakly typed network (TensorRT 10 and earlier) `enabled_precisions` listed
the types the builder was *allowed* to choose from. **TensorRT 11 networks are
strongly typed**: the graph states the precision of every operation and the
builder honours it, so there is nothing left to choose.

The call still succeeds, emits **no warning**, and hands back an FP32 module —
same latency, zero error. A model "converted to FP16" this way was never
converted at all.

## The replacement

```python
torch_tensorrt.compile(
    exported.module(), arg_inputs=inputs, min_block_size=1,
    enable_autocast=True,
    autocast_low_precision_type=torch.float16,   # or torch.bfloat16
)
```

| | latency | output dtype | max rel err |
| --- | ---: | --- | ---: |
| no autocast (all FP32) | 0.149 ms | float32 | 0.00e+00 |
| autocast, FP16 | 0.131 ms | float16 | 1.76e-03 |
| autocast, BF16 | 0.135 ms | bfloat16 | 5.76e-03 |

BF16 keeps FP32's exponent range but has three fewer mantissa bits than FP16, so
here it is the *less* accurate of the two and no faster. It earns its keep on
models that overflow in FP16 — not this one.

## Keeping part of the graph in FP32

```
autocast FP16                          :  0.132 ms, out float16   max rel err 1.76e-03
+ excluded op: aten.linear             :  0.152 ms, out float16   max rel err 3.07e-04
+ excluded nodes: ^linear$, ^linear_1$ :  0.130 ms, out float16   max rel err 1.76e-03
```

Row 2 — excluding the only expensive operator removes the whole benefit: FP16 is
now *slower* than plain FP32, because all that stays in FP16 is casts and `relu`.
The exclusion list is not free.

Row 3 — identical to row 1, i.e. **the pattern matched nothing**.
`autocast_excluded_nodes` is matched against the node names of the *lowered*
graph, which are not the ones `torch.export` prints, and a pattern that matches
nothing is silently ignored. Prefer `autocast_excluded_ops` unless the exact
lowered names are known.

## Composing with `torch.autocast`

PyTorch's own autocast marks part of the graph at export time; Torch-TensorRT
Autocast handles the rest. They are independent, so outputs can end up with
different dtypes:

```
feature (outside torch.autocast): eager float32   -> TensorRT bfloat16
logit   (inside  torch.autocast): eager float16   -> TensorRT float16
```

## Related

+ Cookbook skill `trt-strong-typing-migration` — the same change at the `INetworkDefinition` and `trtexec` level.
+ [`../TorchCompileBackend/`](../TorchCompileBackend/README.md) — where the compile options come from.
