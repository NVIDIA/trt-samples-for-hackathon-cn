# Save and Load

+ Persist a compiled module, and keep its dynamic shapes alive across the trip.

+ Steps to run.

```shell
python3 main.py
```

See [`../DynamicShapes/`](../DynamicShapes/README.md) for how the dynamic
dimension is declared in the first place; this example is only about surviving
save and load.

## The engine is embedded, not rebuilt

```
file 44 KB, loaded in 0.018 s
  batch=1  : max |before save - after load| = 0.0e+00
  batch=32 : max |before save - after load| = 0.0e+00
```

`torch_tensorrt.save` writes the built engine into an `ExportedProgram`, so
`torch_tensorrt.load(path).module()` deserializes it. Milliseconds instead of
seconds, and **bit-identical** output -- a rebuilt engine would likely differ in
the last ULP.

## The trap: the default call silently makes the model static

```python
torch_tensorrt.save(compiled, path, arg_inputs=[example])   # looks fine, is not
```

`retrace` defaults to **True** and `dynamic_shapes` defaults to **None**. So the
call above re-exports the module against a plain example tensor, and that
re-export pins the batch dimension to that tensor's size. Nothing warns; the
model loads, and returns correct answers *for that one batch size*.

| how it was saved | batches that still work |
| --- | --- |
| `arg_inputs=[tensor]`, no spec, `retrace=True` **(the default)** | `1(x) 4 8(x) 32(x)` |
| `arg_inputs=[tensor]`, `dynamic_shapes=...` | `1 4 8 32` |
| `arg_inputs=[tensor]`, `retrace=False` | `1 4 8 32` |
| `arg_inputs=[Input(min/opt/max)]` | `1 4 8 32` |

`(x)` is `AssertionError: Guard failed: x.size()[0] == 4`. The example tensor is
deliberately batch 4 while `opt_shape` is 8, so a wrong specialization shows up
immediately instead of hiding behind the shape that happens to be tuned for.

**Rule**: with `retrace=True` the shape spec has to come from somewhere -- either
`dynamic_shapes=` or `Input(min/opt/max)` in `arg_inputs`. A bare example tensor
is not enough. With `retrace=False` nothing is re-exported, so the question does
not arise.

## Several dynamic dimensions

One `Input` spec covers batch, height and width at once, which is the main reason
to prefer it over writing a `torch.export.Dim` per axis:

```
(4, 3, 128, 128)  -> (4, 16, 128, 128)    max |before - after| = 0.0e+00
(12, 3, 384, 384) -> (12, 16, 384, 384)   max |before - after| = 0.0e+00
(16, 3, 512, 512) -> (16, 16, 512, 512)   max |before - after| = 0.0e+00
```

## Note

`output_format="torchscript"` does not accept `Input` objects in `arg_inputs`
(`Tracer cannot infer type of Input(...)`); it needs real tensors.
