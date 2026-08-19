# RefitObserver

+ `IRefitterObserver` — record at build time **how** every refittable engine weight is produced from
  the ONNX graph, then replay that recording at deploy time to refit with **no ONNX parser**.

+ Added in **TensorRT 11.2** (`NvOnnxParser.h`), **C++ only** — there is no Python binding, and none
  in 11.3 either (the 11.3 `cp312` wheel contains neither `RefitterObserver` nor
  `set_refit_observer`). `04-Feature/Refit` covers the ordinary Python refit workflow.

## Running

```bash
python3 export.py    # build the ONNX models and their weight files
make build
./main.exe
```

The system TensorRT here is **11.1.0.106**, which predates the API. The Makefile therefore builds
against an unpacked tarball given by `TRT_OBSERVER_PATH` (default `/work/trt/TensorRT-11.3.0.93`),
and `LD_LIBRARY_PATH` must include its `lib/` — the builder `dlopen`s
`libnvinfer_builder_resource_sm100.so` at run time, which `-Wl,-rpath` does not cover.

## The problem

Build a refittable engine from ONNX and ask what can be refitted:

```
getAllWeights() reports 7 refittable weights:
  conv.weight  conv.bias  const_out  tmp_refittable_weight  gain.double
  tmp_refittable_weight_0  tmp_refittable_weight_1
```

**Three of those names appear nowhere in the ONNX file.** `IRefitter::setNamedWeights` wants
`tmp_refittable_weight`, and nothing tells you which initializers feed it or what arithmetic turns
them into it. The usual workaround is to keep the `.onnx` around and re-run `IParserRefitter` at
deploy time, which means shipping the model and linking protobuf.

## What the observer emits

One `RefitRecord` per refittable weight, during `refitFromFile`:

| engine weight | transform | count | epsilon | sources |
| ------------- | --------- | ----- | ------- | ------- |
| `conv.weight` | `kIDENTITY` | 144 | — | `conv.weight` |
| `conv.bias` | `kIDENTITY` | 4 | — | `conv.bias` |
| `gain.double` | `kDOUBLE_TO_FLOAT` | 4 | — | `gain.double` |
| `tmp_refittable_weight` | `kBATCH_NORM_FOLD_SCALE` | 4 | 0.001 | `bn.scale bn.bias bn.mean bn.var` |
| `tmp_refittable_weight_0` | `kBATCH_NORM_FOLD_BIAS` | 4 | 0.001 | `bn.scale bn.bias bn.mean bn.var` |
| `const_out` | `kCONSTANT_NODE` | 4 | — | 16 bytes carried **in the record** |
| `tmp_refittable_weight_1` | `kCONSTANT_OF_SHAPE` | 1 | — | 4 bytes carried **in the record** |

That is all six `RefitTransformKind` values on one small model. The two `kCONSTANT*` kinds are the
interesting shape of the API: their values live in a *node attribute*, not an initializer, so there
is nothing to look up by name — the parser hands you the resolved bytes in `RefitRecord::fixedData`.

**The BatchNormalization rows are the ones worth staring at.** TensorRT has no BatchNormalization
layer, so the parser folds four ONNX initializers into two engine weights:

```
combinedScale[i] = scale[i] / sqrt(variance[i] + epsilon)
combinedBias[i]  = bias[i] - mean[i] * combinedScale[i]
```

with an epsilon that came from a node attribute. Reproducing that by hand means re-deriving the
parser's fusion rules and tracking them across releases. The observer just tells you, and the
formulas are documented on the enumerators.

## The payoff

`main.cpp` writes the recording to a **469-byte** `refit-table.txt`. At deploy time it reads only
that table plus a second set of weights, and refits:

```
engine as built      vs reference-v1: max |diff| = 2.384e-07
replaying 7 entries against model-v2.wts (7 blobs), no ONNX parser involved
missing weights after replay: 0
engine after replay  vs reference-v2: max |diff| = 4.768e-07
engine after replay  vs reference-v1: max |diff| = 3.492e+00   (it really moved)
```

The deploy-time dependency is now **the plan and a 469-byte text file**. No `.onnx`, no parser, no
protobuf. The weights arrive in the `.wts` format from
[`02-API/Network/weight_transport.py`](../../02-API/Network/README.md), chosen here precisely
because reading it needs nothing but `std::ifstream`.

## Two traps

+ **`RefitRecord`'s pointers are owned by the parser** and are valid only for the duration of the
  `onRefittableWeight()` call. `TableRecorder` copies every string and every byte; keeping a pointer
  is a use-after-free.
+ **TF32 is on by default.** Before it was cleared, the engine was already `3.5e-04` from
  onnxruntime, which is 700x the error the refit itself introduces and would have hidden it
  completely. Same lesson as [`02-API/Network`](../../02-API/Network/README.md).
