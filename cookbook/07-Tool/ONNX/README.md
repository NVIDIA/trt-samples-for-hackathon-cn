# Onnx

+ An open source format for AI models, both deep learning and traditional ML.

+ Repository [Link](https://github.com/onnx/onnx/tree/main) ·
  [Document](https://onnx.ai/onnx/)

+ Steps to run.

```bash
python3 main.py
```

Measured with **onnx 1.21.0** (opset 26, IR 13) and **TensorRT 11.1.0.106**.

## What this directory is for

The cookbook uses `onnx.helper` and `onnx.checker` in a dozen places already — `../OnnxVisualization/`
alone builds 28 models with them — because building a toy model is what those two are for. So this
example deliberately covers the **other half** of the library: model surgery and interop.
`shape_inference`, `version_converter`, `utils.extract_model`, `compose`, `defs`, `parser`,
`inliner`, `reference` and the metadata fields had **zero** live usage anywhere in the tree.

Everything here is the plain `import onnx` package, which is a different thing from each of its
neighbours:

| Directory                                                    | Package               | Does                                                     |
| ------------------------------------------------------------ | --------------------- | -------------------------------------------------------- |
| here                                                         | `onnx`                | the reference implementation of the *format*             |
| [`../OnnxGraphSurgeon/`](../OnnxGraphSurgeon/README.md)      | `onnx_graphsurgeon`   | a convenient mutable graph API for editing models        |
| [`../Onnxruntime/`](../Onnxruntime/README.md)                | `onnxruntime`         | an *inference engine* for the format                     |
| [`../Polygraphy/`](../Polygraphy/README.md)                  | `polygraphy`          | a debugger that drives all of the above plus TensorRT    |
| [`../OnnxWeightProcess/`](../OnnxWeightProcess/README.md)    | `onnx`                | external-data weights specifically                       |

Each case below is chosen because it changes what TensorRT does, or explains a message TensorRT
prints.

## 1. `check_model()` is weaker than it looks

The default check validates the proto's *structure*. It does not run shape inference, so a graph
whose shapes cannot possibly work is "valid":

```txt
check_model(full_check=False) -> passed
check_model(full_check=True ) -> InferenceError: [ShapeInferenceError] Inference error(s): (op_type:Mul): [ShapeInferenceError] Incompatible dimensions
TensorRT on the same model    -> ok=False, Invalid Node - node_of_b
```

The model is a `Mul` of `[N, 4]` against a `[3]` initializer — no broadcast makes that legal. The
default checker approves it; TensorRT rejects it, from inside `shapeContext.cpp`. **`full_check=True`
finds the same defect before you get to TensorRT, and names the operator instead of a source file.**
This is where "but the ONNX checker said it was fine" bug reports come from.

`check_model` also accepts a **path** instead of a `ModelProto`, which is the only way to check a
model above the 2 GiB protobuf ceiling — such a model cannot be `onnx.load`ed into a proto at all.

## 2. Where `unk__0` comes from

`infer_shapes` propagates shapes forward from the graph inputs. It is *static*, so it cannot look
inside a `Reshape`'s shape tensor. On the cookbook's MNIST model:

```txt
max_pool2d_1   ['nBS', 64, 7, 7]
view           ['unk__0', 3136]
linear         ['unk__0', 1024]
relu_2         ['unk__0', 1024]
softmax        ['nBS', 10]
the Reshape's shape tensor is [-1, 3136]
```

`nBS` survives conv and pool, then **dies at the `-1` and never comes back**. That anonymous
dimension is the `unk__0` you then see in polygraphy and TensorRT output.

`softmax` is back to `nBS` only because the exporter *declared* the graph output `y` as `[nBS, 10]`
— inference treats declared graph I/O as fact, it did not re-derive the symbol. So: a symbol lost
mid-graph is not recoverable by inference alone. If you need it back for an optimization profile,
declare it or rewrite the `Reshape`. (`data_prop=True` does not help here, and was checked.)

## 3. Which opset does this op need?

`onnx.defs` is the op schema registry, and the authoritative answer to a question nothing else in
the toolchain answers. `since_version` is the opset in which the op's *current* definition landed:

```txt
op         latest since_version   revised at opsets
Relu                        14   [1, 6, 13, 14]
Conv                        22   [1, 11, 22]
Softmax                     13   [1, 11, 13]
Resize                      19   [10, 11, 13, 18, 19]
Reshape                     25   [1, 5, 13, 14, 19, 21, 23, 24, 25]
```

When a parser says it does not support an op, it often supports the op *at another opset*. That
churn — `Reshape` has been revised 8 times — is also what the next case runs into.

## 4. Opset conversion goes up, not down

```txt
model-trained.onnx is opset 18
-> opset 26: converted, 12 nodes; TensorRT parses it, 27 layers
-> opset 21: converted, 12 nodes; TensorRT parses it, 27 layers
-> opset 13: Assertion `false` failed: No Adapter From Version $14 for Relu
-> opset 11: Assertion `false` failed: No Adapter From Version $14 for Relu
```

Two things worth taking away:

1. **The downgrade failure is a C++ `assert`, not a Python exception with a plan.** The useful part —
   which op has no adapter — is at the *end* of a message that starts with a path into
   `BaseConverter.h`. Read the tail. The library only ships adapters somebody wrote; downgrading is
   not a supported operation in general, however much a stuck deployment target wants it.
2. **Opsets 18, 21 and 26 all became the same 27 TensorRT layers.** Converting *for* TensorRT is
   almost never the fix. Convert when some other tool in the chain demands a particular version.

## 5. Cutting a subgraph out, in one call

```python
onnx.utils.extract_model(input_file, output_file, ["max_pool2d_1"], ["relu_2"])
```

```txt
model-trained.onnx  : 12 nodes
model_extracted.onnx:  3 nodes ['Reshape', 'Gemm', 'Relu']
kept initializers   : ['gemm1.weight', 'gemm1.bias', 'val_5']
new graph input     : [('max_pool2d_1', ['nBS', 64, 7, 7])]
```

It collected the initializers the cut needs and gave the new input its **inferred** shape, symbol
included; the result passes `full_check` and TensorRT builds it. The two tensor names are the only
thing you have to know — which makes this the fastest way to produce a minimal reproducer for a
TensorRT bug. [`../OnnxGraphSurgeon/08_isolate_subgraph.py`](../OnnxGraphSurgeon/08_isolate_subgraph.py)
does the same thing with the graphsurgeon API, when you need the surrounding graph edited too.

## 6. Merging two models without hand-splicing them

`onnx.compose.merge_models` **refuses ambiguity instead of guessing**, which is the whole reason to
prefer it over editing tensor names yourself:

```txt
naive merge          -> ValueError: Cant merge two graphs with overlapping names. Found repeated edge names: b
after add_prefix     -> [('g1_node', 'Relu'), ('second/g2_node', 'Sigmoid')]
opset 18 + opset 13  -> ValueError: Can't merge two models with different operator set ids for a given domain.
```

Any two independently exported models call their tensors `a`, `b`, `input`, `output`; `add_prefix`
is the fix, and the collision is reported rather than silently resolved. The opset refusal is the
more valuable one: splicing two different-opset models by hand produces a file that loads, passes
the default checker, and is wrong. Compare
[`../OnnxGraphSurgeon/13_merge_two_models.py`](../OnnxGraphSurgeon/13_merge_two_models.py), where
both of those responsibilities are yours.

## 7. A text format and a pure-Python evaluator

`onnx.parser` reads ONNX's own textual IR — far shorter than a page of `make_node` calls, and the
best way to write a reproducer into a bug report:

```txt
<ir_version: 10, opset_import: ["": 18]>
agraph (float[N, 4] X) => (float[N, 4] Y) {
    two = Constant <value_float: float = 2.0> ()
    scaled = Mul(X, two)
    Y = Relu(scaled)
}
```

`onnx.reference.ReferenceEvaluator` then executes it in pure Python. It is slow and it is not what
you deploy, but it is **the specification's answer rather than an implementation's** — a third
opinion that belongs to neither onnxruntime nor TensorRT, which is exactly what you want when those
two disagree. No GPU, no onnxruntime, no extra install.

## 8. Local functions do not need inlining for TensorRT

A local function is a reusable subgraph carried inside the model; `onnx.inliner` expands the call
sites into ordinary nodes. Whether that is a *required* step on the TensorRT path is the question
worth answering, and it is not:

```txt
with functions: 1 function(s), nodes ['local.Scaled', 'local.Scaled']
                TensorRT ok=True, 6 layers
inlined       : 0 function(s), nodes ['Constant', 'Mul', 'Constant', 'Mul']
                TensorRT ok=True, 6 layers
```

**Identical layer counts**: the TensorRT parser already inlines local functions itself. Inline for
the *readers* — Netron, a diff, a tool that predates functions — not for TensorRT.

## 9. The provenance fields

`producer_name`, `producer_version`, `model_version`, `domain`, `doc_string` and the free-form
`metadata_props` key/value list all survive a save/load round trip, and **none of them affects the
engine** — the parser builds the same 27 layers before and after. They are also never validated, so
nothing will fill them in for you. They are the only place to record which script, commit and
quantization recipe produced a `.onnx` that will outlive the terminal it was built in.

`ir_version` and `opset_import` are the two fields in this area that *do* change behaviour; see
case 4.

## Not covered here

+ **External-data weights** (`save_as_external_data` / `load_external_data_for_model`) — has its own
  directory, [`../OnnxWeightProcess/`](../OnnxWeightProcess/README.md).
+ **Building models with `onnx.helper`** — done at length in
  [`../OnnxVisualization/00-ModelZoo/model_zoo.py`](../OnnxVisualization/00-ModelZoo/model_zoo.py)
  (28 models), and again in `../Polygraphy/` and `../../08-Advance/TensorRTGraphSurgeon/`.
+ **`onnx.hub`** — downloads models from a model zoo; the cookbook keeps its models in
  [`../../00-Data/`](../../00-Data/README.md) on purpose.
+ **`onnxoptimizer` / `onnxsim` / `onnxslim`** — separate packages, not part of `onnx`. Graph
  simplification in the cookbook goes through polygraphy's `fold_constants`, see
  [`../OnnxGraphSurgeon/07_shape_operation_and_simplify.py`](../OnnxGraphSurgeon/07_shape_operation_and_simplify.py).
