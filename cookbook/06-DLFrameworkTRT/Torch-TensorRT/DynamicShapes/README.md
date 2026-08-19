# Dynamic Shapes

+ One engine that serves a range of input shapes, instead of one engine per shape.

+ Steps to run.

```shell
python3 main.py
```

## Three ways to declare a dynamic dimension

| Where | API |
| --- | --- |
| JIT | `torch._dynamo.mark_dynamic(x, index=0, min=1, max=32)` then `torch.compile(backend="tensorrt")` |
| AOT | `torch_tensorrt.Input(min_shape=..., opt_shape=..., max_shape=...)` |
| AOT | `torch.export.Dim("batch", min=1, max=32)` in `torch.export.export(dynamic_shapes=...)` |

All three produce **bit-identical** results across the whole declared range
(`max |eager - TensorRT| = 0.0` at batch 1 / 4 / 17 / 32).

`opt_shape` is the one TensorRT tunes its tactics for; the engine still runs at
the other sizes, just less well tuned.

## Where they differ: outside the declared range

This is the practical difference, and it is not a matter of taste.

```
AOT batch=33: RuntimeError from the runtime, engine profile is a hard bound
JIT batch=33: accepted after 7.29 s -- dynamo re-traced and rebuilt
JIT batch=64: 0.05 s -- the rebuilt engine already covers it
```

AOT hands TensorRT an optimization profile with hard bounds, so an out-of-range
input is a `setInputShape` failure. JIT keeps dynamo in the loop, so the same
input is just another guard failure: it recompiles, costs seconds, and continues.

Neither is "correct". A deployed service usually wants AOT's loud, predictable
failure; an interactive workload usually prefers JIT's stall.

## Why bother at all

`case_static_baseline` compiles with no dynamic dimension, then feeds a different
batch size. The class token `expand`, the `cat` and the `reshape` all bake the
traced batch in, so it is rejected outright -- not slow, rejected.

## Trap: `min_block_size` silently leaves the model in eager

**`min_block_size` defaults to 5.** A sub-graph with fewer operators is not
converted at all -- no warning, no exception, and the module still returns
correct results. This model lowers to three operators (`cat`, `linear`,
`reshape`):

```
min_block_size=5: static engines=0  0.155 ms | dynamic engines=1  0.115 ms   <- default
min_block_size=1: static engines=1  0.114 ms | dynamic engines=1  0.113 ms
```

At the default the **static** model never reaches TensorRT, while the dynamic one
does (its symbolic-shape arithmetic adds enough nodes to clear the threshold). So
a "static vs dynamic" comparison at the default is really "eager vs TensorRT" --
the first draft of this example measured the dynamic engine as *faster* than the
static one for exactly that reason.

Every case therefore prints `TensorRT sub-modules: N`. If that is 0, nothing was
compiled and any measurement taken against it is meaningless.

## What the shape range actually costs

Once both sides really are engines:

```
at batch=4 (= opt_shape): static 0.109 ms, dynamic 0.111 ms (1.02x)
at batch=1             : dynamic 0.105 ms
at batch=32            : dynamic 0.127 ms
```

Essentially free at `opt_shape`, and the drift away from it is small here.
