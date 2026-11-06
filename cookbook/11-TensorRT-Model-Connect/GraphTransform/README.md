# GraphTransform

+ Change a TensorRT network built by code you do not own, at the last moment before it is compiled.

```bash
python3 main.py
```

Adapted from `core/builder/tensorrt_model_connect/graph_transform.py` in
[TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect) (Apache-2.0).

## The problem

You import a builder — a library's, a colleague's, `build_mnist_network_trt` here — and you need the
network it produces to be slightly different. One extra output for debugging, one layer's precision
pinned, one metadata field set. The usual answers are to fork the builder or to add a parameter to
it, and both of those make every caller's problem your problem.

## The hook

`trt.Builder` is wrapped by an object that forwards everything through `__getattr__` and overrides
exactly one method:

```python
def build_serialized_network(self, network, config):
    self._transform(network, next(self._index_iterator))
    return self._builder.build_serialized_network(network, config)
```

That line runs at the only moment that works. Earlier, the network is still being assembled and the
layer you want may not exist; later, there is nothing but engine bytes and `INetworkDefinition`
changes mean nothing.

Two ways in, because there are two situations:

| Situation | Call |
| --- | --- |
| You can reach the builder object | `tw.builder = wrap_builder(tw.builder)` |
| The builder is created somewhere you cannot reach | `trt.Builder = BuilderFactory(trt.Builder)` |

`TRTWrapperV1` needs both at different times: it creates its builder in `__init__`, so a test that
wants to hook a wrapper it has already constructed uses the first, and `case_patch_the_builder_class`
shows the second.

## Scoping is not decoration

The active transform lives in a `ContextVar`, not a module global:

+ two threads building different models each see their own transform;
+ an exception inside the `with` block cannot leak a transform into the next build —
  `case_scope_is_the_whole_contract` raises on purpose and asserts the scope is clear afterwards;
+ `graph_transform(None)` is a documented no-op, so a caller with an optional transform needs no
  branch.

`engine_index` comes from a counter owned by one scope. A request that builds a prefill engine and a
decode engine sees `0` then `1`, and the next request starts at `0` again. `case_engine_index_separates_builds`
asserts exactly that.

## What the example measures

`case_expose_an_internal_tensor` marks the output of `Activation1` as an engine output without
editing `build_mnist_network_trt`:

```txt
    Without a transform : ['x', 'y', 'z']
    transform(engine_index=0) marked 'Activation1_output' as an output, dtype DataType.FLOAT
    With    a transform : ['x', 'y', 'z', '(Unnamed Layer* 1) [Activation]_output']
```

**Two details in that output are worth more than the feature itself.**

+ **The name in the network is not the name on the binding.** `ITensor.name` reads
  `Activation1_output`, derived from `layer.name = "Activation1"` which the upstream builder set.
  The engine binding is `(Unnamed Layer* 1) [Activation]_output` — the default the tensor had before
  the layer was renamed. Bind by what the *engine* reports, never by what the network said.
+ **There is no dtype to set.** The network is strongly typed, so `ITensor.dtype` is read-only;
  assigning to it raises `AttributeError: property of 'ITensor' object has no setter` on
  TensorRT 11.3. A marked output keeps the type the graph gives it.

## Where this is not the answer

If you own the builder, add a parameter. This exists for the case where you do not, and it is a
sharp tool: a transform that renames a tensor, removes a layer, or marks an output that something
downstream does not expect will produce an engine that builds cleanly and behaves differently. The
upstream project runs its transform twice — once when deciding which nodes to claim and once when
building — precisely so the two views cannot drift. Our `case_expose_an_internal_tensor` asserts on
the binding list for the same reason.

Measured on TensorRT 11.3.0.99, B200, 2026-10-12.
