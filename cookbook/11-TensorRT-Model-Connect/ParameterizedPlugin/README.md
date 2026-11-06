# ParameterizedPlugin

+ One plugin that hosts any registered kernel, selected by a string at build time.

```bash
python3 main.py
```

The idea is from `core/runtime/byok/` plus `core/builder/tensorrt_model_connect/byok.py` in
[TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect) (Apache-2.0), where one
`TvmFfiKernel` plugin takes a `kernel_name` string and a JSON `shape_spec` as plugin fields and
dispatches to a TVM-FFI function. Nothing is copied: theirs is 976 lines of C++
`IPluginV2DynamicExt`, this is Python on `tensorrt.plugin`.

## The pattern

Compare with [`../../05-Plugin/QuickDeployablePlugin/`](../../05-Plugin/QuickDeployablePlugin/README.md),
which registers one plugin per operation — `sample::elemwise_add_plugin`,
`sample::elemwise_add_plugin_`, and so on. Here `cookbook::byok` is registered **once** and the
operation is an attribute:

```python
@register_kernel("cookbook.max_of_two")
def max_of_two(a, b, out, extra_args, stream):
    torch.maximum(a, b, out=out)
```

That decorator is the entire cost of adding an operation. No new plugin class, no new
`IPluginCreator`, no second entry in the plugin registry, no rebuild.

```txt
    Registered kernels: ['cookbook.residual_add', 'cookbook.row_sum', 'cookbook.scaled_add']
    residual_add -> shape (2, 3, 8), max abs diff 0.000e+00
    scaled_add   -> shape (2, 3, 8), max abs diff 0.000e+00 (alpha came from the spec)
    row_sum      -> shape (2, 3), max abs diff 4.768e-07 (rank dropped by 1)
```

## Why the spec is JSON and not plugin attributes

A plugin's attribute list is fixed when the plugin is registered, but the knobs are per kernel:
`scaled_add` wants `alpha`, `row_sum` wants nothing, the next kernel will want something else. One
opaque string keeps the plugin signature stable while the kernels vary. The upstream project made
the same call for the same reason.

## Two things that have to be declared, not discovered

**The output shape.** It cannot be read off a kernel that has not run yet, and shape inference
happens long before enqueue. `register_kernel(..., rank_delta=-1)` declares that `row_sum` removes
one trailing dimension, and the descriptor builds the output shape from **symbolic** expressions,
because at that point the batch dimension is still `-1`:

```python
shape_expr = inp0.shape_expr
kept = tuple(shape_expr[i] for i in range(len(shape_expr) + rank_delta))
return trtp.from_shape_expr(kept, dtype=inp0.dtype)
```

**The arity.** `trtp.register` takes the input count from the decorated function's signature, so
every kernel here takes exactly two inputs. The upstream escapes this by carrying `num_inputs` /
`num_outputs` in its JSON and calling `add_plugin_v2` with a variable-length list; the typed Python
API has no equivalent. If you need variable arity, that is the reason to drop to C++.

## Failing at the right time

`make_spec` looks the kernel up when the spec is created — at network-build time:

```txt
    "Kernel 'cookbook.does_not_exist' is not registered; known kernels: ['cookbook.max_of_two', ...]"
```

The alternative is a `KeyError` inside `enqueue`, in the middle of inference, where a plugin
exception is far more expensive to diagnose. Validate attributes where the user can still fix them.

## Scope

The kernels here are demonstrations of the dispatch mechanism, not of kernel writing: two are
Triton, two are `torch` one-liners. For kernel authoring see
[`../../05-Plugin/`](../../05-Plugin/README.md). `tensorrt.plugin` still prints
`Functionality provided through tensorrt.plugin module is experimental.` on import.

Measured on TensorRT 11.3.0.99, B200, 2026-10-12.
