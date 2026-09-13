# Mutable Module

+ A compiled module that follows its source model's weights instead of going stale.

+ Steps to run.

```shell
python3 main.py
```

Only the self-contained half of the upstream example is covered; the Stable
Diffusion / LoRA pipeline needs `diffusers` and gated downloads.

## The problem

`torch_tensorrt.compile` copies the weights into the engine. Change the source
`nn.Module` afterwards and:

```
compiled output changed at all : False
still agrees with eager        : False, max |eager - TensorRT| = 2.495
```

The compiled module keeps answering with the weights it was built from. No
exception, no warning — it is just quietly wrong from the moment the source model
is touched.

## The fix

```python
mutable = torch_trt.MutableTorchTensorRTModule(model, immutable_weights=False, min_block_size=1)
mutable(*inputs)                        # lazy: this is what compiles
mutable.load_state_dict(other.state_dict())   # refits the existing engine
```

```
first call compiles      :  10.06 s
load_state_dict + refit  :   1.78 s (5.6x cheaper than compiling)
max |eager - TensorRT|   : 1.68e-07
```

`immutable_weights=False` is required — the same precondition as
[`../EngineCaching/`](../EngineCaching/README.md), and for the same reason: a
refit needs a refittable engine.

## Save and load

```
pickle  150.6 MB, loaded in 0.70 s
max |before save - after load| = 0.00e+00
```

`MutableTorchTensorRTModule.save` / `.load` is **not** the same mechanism as
`torch_tensorrt.save` (see [`../SaveLoad/`](../SaveLoad/README.md)): it preserves
the mutable wrapper, so the reloaded object can still be refitted. The price is
size — the pickle carries the engine *and* the PyTorch weights needed to refit
from.

## Which refit mechanism to use

| goal | use |
| --- | --- |
| swap weights on a live module | `MutableTorchTensorRTModule` (here) |
| skip rebuilding across processes / weight sets | [`../EngineCaching/`](../EngineCaching/README.md) |
| ship one immutable artifact | [`../SaveLoad/`](../SaveLoad/README.md) |

All three ultimately refit; they differ in what holds the engine and how long it
lives.
