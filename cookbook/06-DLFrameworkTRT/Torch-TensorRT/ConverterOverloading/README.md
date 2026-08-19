# Converter Overloading

+ Replace Torch-TensorRT's lowering of one operator with your own.

+ Steps to run.

```shell
python3 main.py
```

## The trap: `STANDARD` priority does not override anything

```
built-in converter      : calls {'standard': 0, 'high': 0}
registered at STANDARD  : calls {'standard': 0, 'high': 0}   <- registered, never called
registered at HIGH      : calls {'standard': 0, 'high': 1}   <- this one wins
```

`ConverterPriority.STANDARD` **appends** to the candidate list for the target
operator. The built-in `gelu` converter is already in that list and its validator
passes, so it is chosen first and yours is never consulted.

Nothing warns: the registration succeeds, the compile succeeds, the results are
correct — the custom code simply never ran. **To override an operator that
Torch-TensorRT already handles, `priority=ConverterPriority.HIGH` is required.**

## Writing one

```python
@dynamo_tensorrt_converter(
    torch.ops.aten.gelu.default,                 # target ATen op
    capability_validator=lambda node, settings:  # per-node, before partitioning
        node.kwargs.get("approximate") == "tanh",
    supports_dynamic_shapes=True,                # False disables it under dynamic shapes
    priority=ConverterPriority.HIGH,             # see above
)
def convert(ctx, target, args, kwargs, name):
    ...  # return trt.ITensor built with impl.* helpers or ctx.net directly
```

`impl.*` (`torch_tensorrt.dynamo.conversion.impl`) wraps the raw TensorRT API so
a converter does not have to hand-build every layer; both styles compose.

## `capability_validator` gates per node

```
erf mode, validator rejects : calls {'standard': 0, 'high': 0}
```

The validator runs against each instance of the target op before partitioning.
Returning `False` hands that node to the next candidate — here the built-in
converter, so nothing falls back to PyTorch. That is how a converter that covers
only part of an operator's schema is written: claim the cases you handle, let the
rest fall through.

## Checked against eager, not against TensorRT

Every case compares to **eager PyTorch** (`max |eager - TensorRT| = 0.00e+00`),
not to the built-in TensorRT path. A custom converter that merely reproduces
TensorRT's own answer proves much less than one that reproduces PyTorch's.

Note the counter rather than a `print` inside the converter: it lets each case
`assert` which converter ran instead of asking the reader to scan the log.

## Related

+ [`../TorchCompileBackend/`](../TorchCompileBackend/README.md) — `torch_executed_ops`, the blunt alternative when the goal is just to avoid a converter.
