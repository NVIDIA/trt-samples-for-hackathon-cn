# TensorRT Graph Surgeon

+ Edit a parsed network at the **`INetworkDefinition`** level, between the ONNX parser and the builder.

Then build and run it.

+ Steps to run.

```bash
python3 main.py
```

ONNX-GraphSurgeon edits the ONNX *before* parsing; this edits what the parser *produced*. Use it
when the thing to change only exists after parsing (a layer the parser synthesised), or when the
ONNX is not yours to modify.

All numbers re-measured on **B200** 2026-09-06 (TensorRT 11.1.0.106, 1965 MHz), with
`x -> Add(1.0) -> Relu -> Mul(2.0) -> y` on a `[8, 512, 512]` tensor.

## The whole toolkit

| Need | API |
| ---- | --- |
| walk | `network.num_layers`, `network.get_layer(i)`, `layer.type` / `name` / `get_input` / `get_output` |
| add | any `network.add_*`, exactly as when building by hand |
| rewire | `layer.set_input(index, tensor)` |
| move the boundary | `network.mark_output` / `unmark_output` |
| delete | **does not exist** — see below |

## The cases

### 1. What the parser actually produced

3 ONNX nodes become **7 TensorRT layers**:

```txt
0: one                    CONSTANT     [] -> ['one_output']
1: ONNXTRT_Broadcast      SHUFFLE      ['one_output'] -> ['ONNXTRT_Broadcast_output']
2: node_add               ELEMENTWISE  ['x', 'ONNXTRT_Broadcast_output'] -> ['a']
3: node_relu              ACTIVATION   ['a'] -> ['b']
4: two                    CONSTANT     [] -> ['two_output']
5: ONNXTRT_Broadcast_1    SHUFFLE      ['two_output'] -> ['ONNXTRT_Broadcast_1_output']
6: node_mul               ELEMENTWISE  ['b', 'ONNXTRT_Broadcast_1_output'] -> ['y']
```

Two things make the rest possible: **the ONNX node names survive** (`node_add`, `node_relu`,
`node_mul`), so a layer can be found by name; and the extra `CONSTANT` / `SHUFFLE` layers are the
parser broadcasting the scalar initializers, which is the kind of structure that only exists on
this side of the parser. `../../07-Tool/Polygraphy/More/11-NetworkAsOnnxLike/` measures the same
effect on a real model (12 ONNX nodes -> 27 TensorRT layers).

### 2. Append a layer

`unmark_output(y)`, add a `NEG` consuming `y`, `mark_output(y_negative)`. The old output tensor
simply goes back to being an ordinary tensor. Forgetting the `unmark_output` leaves the network
with two outputs rather than an error.

### 3. Delete a layer — by not using it

There is no `remove_layer`. The `Relu` is deleted by pointing its consumer at its producer:

```python
add_output = find_layer(network, "node_add").get_output(0)
find_layer(network, "node_mul").set_input(0, add_output)   # was the Relu's output
```

`network.num_layers` stays **7** afterwards: the layer object is still there, just unused, and it
is the *builder* that drops what nothing reads. The proof that it is really gone is in the numbers
— negative values (`-4.0`) appear where the `Relu` used to clamp at zero.

### 4. Replace a layer with your own plugin

The requested case, and the interesting direction: **the ONNX already runs fine in TensorRT**, and
the plugin is wanted anyway (a fused variant, a numerical convention, a kernel the team owns).
Parsing first and swapping afterwards avoids touching the ONNX and avoids the custom-op dance of
[`../../05-Plugin/ONNXParserWithPlugin/`](../../05-Plugin/ONNXParserWithPlugin/README.md):

```python
layer = network.add_plugin_v3([network.get_input(0)], [], plugin)   # AddScalar, scalar = 1.0
find_layer(network, "node_relu").set_input(0, layer.get_output(0))  # node_add is now unused
```

| Engine | layers | kernel time |
| ------ | ------ | ----------- |
| as parsed | 1 | 0.0050 ms |
| `Add` -> `AddScalar` plugin | 2 | 0.0092 ms |

The output is **bit-identical**, and the replacement costs **1.84x** (it was 2.13x on H100). That is not plugin overhead:
TensorRT had folded `Add + Relu + Mul` into a single kernel, and a plugin cannot be fused into it,
so one pass over memory became two — on a bandwidth-bound chain that lands close to 2x. Replace
a *fusable* op only when the plugin brings more than it costs.

The timing deliberately calls `execute_async_v3` directly instead of `TRTWrapperV1.infer`: the
host<->device copies in `infer` are ~150x the kernel time here and would hide the whole effect.

## Two lifetime traps, both found the hard way

+ **Query the plugin registry before you start releasing engines.** With the plugin library loaded,
  building an engine and then letting it go (rebinding the variable is enough) makes the next
  `trt.get_plugin_registry().get_creator(...)` **segfault** — reproducibly, with no Python
  traceback, inside `libnvinfer`. `main.py` therefore loads the library and builds the plugin
  object up front and keeps both wrappers alive. Measured on TensorRT 11.1.0.106; the root cause
  is not established here, only the trigger and the way around it.
+ Only tensors that are still reachable from a network output survive the build. After a rewire,
  double-check the output values, not the layer count — the count does not change.

## Related

+ [`../../05-Plugin/BasicExample/`](../../05-Plugin/BasicExample/README.md) — the `AddScalar`
  plugin used here, and how to write one.
+ [`../../05-Plugin/ONNXParserWithPlugin/`](../../05-Plugin/ONNXParserWithPlugin/README.md) — the
  other way to get a plugin into a parsed model: a custom op in the ONNX itself.
+ [`../../07-Tool/OnnxGraphSurgeon/`](../../07-Tool/OnnxGraphSurgeon/README.md) — surgery on the
  ONNX side, before the parser.
+ [`../../07-Tool/Polygraphy/More/13-PerLayerPrecision/`](../../07-Tool/Polygraphy/More/13-PerLayerPrecision/README.md)
  — `PostprocessNetwork`, which is Polygraphy's hook for exactly this kind of edit.
