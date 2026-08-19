# Building a network by hand

+ Fill an empty TensorRT network with raw API calls, then hand it back to Polygraphy.

+ Steps to run.

```bash
python3 main.py
```

## The pattern

```python
@func.extend(CreateNetwork())
def make(builder, network):
    tensor = network.add_input(name="x", shape=SHAPE, dtype=trt.float32)
    one = network.add_constant(shape=SHAPE, weights=np.ones(SHAPE, dtype=np.float32)).get_output(0)
    output = network.add_elementwise(tensor, one, op=trt.ElementWiseOperation.SUM).get_output(0)
    output.name = "y"
    network.mark_output(output)

with TrtRunner(EngineFromNetwork(make)) as runner:
    ...
```

`CreateNetwork()` gives an empty network, `extend` fills it, and everything after
that is ordinary Polygraphy. `02-API/` builds networks the same way without
Polygraphy — the difference is only who owns the builder and the runner.

## Precision is declared here, not requested from the builder

```
CreateNetwork() -> 0 layers, STRONGLY_TYPED = True

declared float32  -> engine output dtype float32, exact match = True
declared float16  -> engine output dtype float16, exact match = True
```

**Strongly typed is the default and there is no flag to turn it off**, so the
`dtype` arguments are commitments rather than hints. This is the constructive
half of the story the sibling examples only show the negative side of:

+ `CreateConfig(fp16=True)` raises — [`../02-ComparingBackends/`](../02-ComparingBackends/README.md)
+ `trt.BuilderFlag.FP16` no longer exists — [`../04-ExtendInterop/`](../04-ExtendInterop/README.md)

because the answer moved into `add_input(dtype=...)` and the weights' dtype. No
config is touched above.

## A type error is a build error

```
bfloat16 input + float16 weights:
[E] ElementWiseOperation SUM must have same input types. But they are of types BFloat16 and Half.
PolygraphyException: Invalid Engine. Please ensure the engine was built correctly
```

A weakly typed builder used to paper over this by inserting a cast. Here the
engine simply does not build, and the message names the layer and both types.
That is the trade strong typing makes: less convenience, no silent casts.

## Related

+ [`../04-ExtendInterop/`](../04-ExtendInterop/README.md) — the `func.extend` mechanism this relies on.
+ `02-API/` — the same network building without Polygraphy.
+ Cookbook skill `trt-strong-typing-migration`.
