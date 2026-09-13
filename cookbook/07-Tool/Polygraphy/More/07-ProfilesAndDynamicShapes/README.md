# Optimization profiles and dynamic shapes

+ Several profiles in one engine, and whether choosing the right one matters.

+ Steps to run.

```bash
python3 main.py
```

## One engine, three profiles

```python
profiles = [
    Profile().add("x", min=(1, 1, 28, 28),  opt=(1, 1, 28, 28),  max=(1, 1, 28, 28)),   # latency
    Profile().add("x", min=(1, 1, 28, 28),  opt=(4, 1, 28, 28),  max=(32, 1, 28, 28)),  # dynamic
    Profile().add("x", min=(64, 1, 28, 28), opt=(64, 1, 28, 28), max=(64, 1, 28, 28)),  # offline
]
engine = engine_from_network(network_from_onnx_path(onnx_file), config=CreateConfig(profiles=profiles))
```

```
engine holds 3 optimization profile(s)
```

`min == opt == max` pins a profile to one shape, which is what a latency-critical
or an offline path usually wants. The weights are stored once regardless of how
many profiles the engine carries.

## A profile is a hard range, per runner

```
0: pinned to batch 1      : batches 1 4(x) 32(x) 64(x)
1: dynamic 1..32, opt 4   : batches 1 4 32 64(x)
2: pinned to batch 64     : batches 1(x) 4(x) 32(x) 64
```

`TrtRunner(engine, optimization_profile=N)` binds a runner to one profile. A
runner on profile 0 cannot be handed batch 4 **even though the engine supports it
through profile 1** — the range that applies is the selected profile's, not the
engine's union.

## Does choosing well cost anything?

```
batch 1 via 0: pinned to batch 1      :  0.601 ms
batch 1 via 1: dynamic 1..32, opt 4   :  0.607 ms
dynamic profile is 1.01x the pinned one at its worst-case shape
```

**On this model, essentially nothing.** Upstream builds three profiles and stops
there; the advice to "pick the profile matching your shape" is only advice until
somebody measures it. Here the honest answer is that a small MNIST network is not
tactic-sensitive enough for the tuning to show. On a large model it can matter —
but that is a claim to verify per model, not to inherit.

## Different profiles, same maths

```
profile 1 vs profile 0: max |diff| = 7.451e-09
```

Profiles change which tactics TensorRT picks, not what the network computes. That
7.45e-09 is the same FP32 rounding measured in
[`../02-ComparingBackends/`](../02-ComparingBackends/README.md).

## Related

+ `06-DLFrameworkTRT/Torch-TensorRT/DynamicShapes/` — the same topic from the PyTorch side, where an out-of-range input either raises (AOT) or silently recompiles (JIT).
+ [`../01-LazyVsImmediate/`](../01-LazyVsImmediate/README.md) — this example mixes both API styles, which is normal and explained there.
