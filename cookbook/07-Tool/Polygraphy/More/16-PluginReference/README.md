# Plugin reference implementations

+ Check a TensorRT plugin against a CPU reference with `PluginRefRunner`.

+ Steps to run.

```bash
python3 main.py
```

The plugin comes from [`05-Plugin/ONNXParserWithPlugin/`](../../../../05-Plugin/ONNXParserWithPlugin/README.md);
`main.py` runs its `make build` if `AddScalarPlugin.so` is not there yet.

## Neither word in the name is quite right

`PluginRefRunner` walks an ONNX-GraphSurgeon graph and evaluates every node with
a NumPy function from `OP_REGISTRY`. It never loads a plugin, and the registry
ships with three ops:

```
ops with a reference implementation: ['Identity', 'InstanceNormalization', 'MeanVarianceNormalization']
the AddScalar model is one node : ['AddScalar']
running it unregistered -> PolygraphyException: Op: AddScalar does not have a reference implementation registered!
a normal MNIST model uses       : ['ArgMax', 'Conv', 'Gemm', 'MaxPool', 'Relu', 'Reshape', 'Softmax']
running that              -> PolygraphyException: Op: Conv does not have a reference implementation registered!
```

It evaluates the *whole* graph, so one unregistered node stops it — and `Conv`
is as unsupported as a custom op. This is not a runner you point at a model. It
is one you point at a single op.

## Loading the plugin is where it breaks

`LoadPlugins` calls `ctypes.CDLL`. That was how plugins loaded back when
`REGISTER_TENSORRT_PLUGIN` ran at static-initialisation time; an
`IPluginCreatorV3One` is picked up by `IPluginRegistry::loadLibrary` instead:

```
creators registered at startup            : 16, AddScalar present: False
after Polygraphy's LoadPlugins (ctypes)   : 16, AddScalar present: False
after trt.init_libnvinfer_plugins          : 44, AddScalar present: False
after registry.load_library                : 45, AddScalar present: True
```

Only the third works, and the first two report success. `LoadPlugins` even logs
`Loading plugin library: ...` at INFO. What you get instead is the ONNX parser,
two layers away:

```
[6] creator && "Plugin not found, are the plugin name, version, and namespace correct?"
```

which sends you off to check the name, version and namespace — all of which are
fine.

The CLI's `--plugins` flag goes through the same `LoadPlugins`, so it cannot load
this plugin either:

```
`polygraphy run --plugins ...` exit code 1: [6] creator && "Plugin not found, are the plugin name, version, and namespace correct?"
```

The working spelling is the one
[`tensorrt_cookbook.utils_plugin.load_plugin_files`](../../../../tensorrt_cookbook/utils_plugin.py)
already uses, where the `ctypes` path is commented out and labelled
`[Deprecated]`:

```python
trt.get_plugin_registry().load_library(str(plugin_file))
```

## The workflow it exists for

```python
from polygraphy.backend.pluginref.references import register   # not re-exported

@register("AddScalar")
def run_add_scalar(attrs, x):
    return [x + attrs["scalar"]]
```

`register` is not in `polygraphy.backend.pluginref`'s exports, so the one
function you need to use the feature at all is the one that looks private. The
decorated function takes `(attrs, *inputs)` — constant inputs arrive as NumPy
arrays already — and returns a **list**, one entry per node output.

```
registry after registering: ['AddScalar', 'Identity', 'InstanceNormalization', 'MeanVarianceNormalization']
trt-runner        : [1. 2. 3. 4.]
pluginref-runner  : [1. 2. 3. 4.]
accuracy comparison passes: True
```

## A passing comparison proves less than it looks

The mismatch this catches most often is an ignored attribute, not arithmetic.
A reference that hard-codes `1.0` agrees perfectly with a model whose `scalar`
attribute happens to be `1.0`:

```
hard-coded 1.0, model attribute is 1.0 -> passes: True
same reference, model attribute is 5.0 -> passes: False
plugin says [5. 6. 7. 8.], reference says [1. 2. 3. 4.], max |diff| = 4.0
```

Nothing about the reference changed between those two lines. It was wrong both
times; only the second model could see it. Also note `OP_REGISTRY` is a plain
dict — re-registering an op replaces the previous entry with no warning.

## Cutting the plugin out of a real model

Whole-graph evaluation means a model with one custom op among ordinary ones has
to be reduced first. `Mul` has no reference either:

```
full graph nodes: ['Mul', 'AddScalar']
PluginRefRunner on the full graph -> Op: Mul does not have a reference implementation registered!
subgraph nodes  : ['AddScalar'], inputs ['scaled']
plugin vs reference on the isolated node: True
and the full model still runs in TensorRT: [1. 3. 5. 7.]
```

`onnx_graphsurgeon` does the cut: reassign `inputs`/`outputs`, then `cleanup()`.
The trap is that `graph.copy()` makes **new tensor objects**, so a `Variable` held
from the original graph is not the one in the copy — look it up by name with
`copy.tensors()` or `cleanup()` raises `Encountered a node not in the graph`.

The CLI spelling of the whole thing is:

```bash
polygraphy run model.onnx --trt --pluginref
```

which has the same three-op limitation, plus the `--plugins` problem above.
There is no flag for registering a fourth op — `--gen-script` and edit the
result, or ship the registration in an extension module (see
[`../19-CustomBackend/`](../19-CustomBackend/README.md)).
