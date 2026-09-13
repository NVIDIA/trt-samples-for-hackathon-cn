# Per-layer precision and per-tensor formats

+ Polygraphy's four network post-processing loaders, against a strongly-typed
  network.

+ Steps to run.

```bash
python3 main.py
```

## Everything is strongly typed now

`create_network()` with no flags at all already carries `STRONGLY_TYPED`, and
there is no flag to turn it off. The weakly-typed network — where TensorRT was
free to pick a layer's precision and insert casts — is gone.

`NetworkFromOnnxPath` still accepts `strongly_typed=False`, and ignores it:

```
trt.Builder.create_network() with no flags -> STRONGLY_TYPED = True
NetworkFromOnnxPath(strongly_typed=True ) -> STRONGLY_TYPED = True
NetworkFromOnnxPath(strongly_typed=False) -> STRONGLY_TYPED = True
NetworkFromOnnxPath(strongly_typed=None ) -> STRONGLY_TYPED = True
```

No warning, no error, the flag comes back set. That premise decides what the four
loaders do.

## Four loaders, three outcomes

| Loader | On TensorRT 11 |
| --- | --- |
| `SetLayerPrecisions` | clean refusal — `ILayer.precision` was removed |
| `SetTensorDatatypes` | raw pybind error — `ITensor.dtype` has no setter |
| `SetTensorFormats` | works — a format is a layout, not a type |
| `PostprocessNetwork` | works — the general escape hatch |

`SetLayerPrecisions` fails the good way. `ILayer.precision` / `.precision_is_set`
went, together with the `OBEY_PRECISION_CONSTRAINTS` and
`PREFER_PRECISION_CONSTRAINTS` flags that gave them meaning, and Polygraphy
checks for the attribute and names the version:

```
trt.ILayer.precision exists: False
BuilderFlag.*PRECISION_CONSTRAINTS: none left
SetLayerPrecisions: PolygraphyException: layer precision in SetLayerPrecisions is not available on TensorRT version 11.1.0.106.
```

Compare `Calibrator` in [`../06-Int8IsNowExplicit/`](../06-Int8IsNowExplicit/README.md)
and `TacticRecorder` in [`../12-TacticsAndReproducibility/`](../12-TacticsAndReproducibility/README.md),
which construct happily and fail much later.

`SetTensorDatatypes` fails the bad way:

```
SetTensorDatatypes: AttributeError: property of 'ITensor' object has no setter
```

That is pybind's setter check, two layers below the cause. It says nothing about
strong typing, nothing about which tensor, nothing about the version. The
replacement is to declare the type at `add_input` —
[`../05-BuildNetworkByHand/`](../05-BuildNetworkByHand/README.md).

## Format is the one knob left, and it is boxed in

`SetTensorFormats` works, and the result is visible on the built engine:

```
format LINEAR : built   engine reports x as TensorFormat.LINEAR, dtype DataType.FLOAT
format CHW32  : built   engine reports x as TensorFormat.CHW32,  dtype DataType.FLOAT
format HWC    : built   engine reports x as TensorFormat.HWC,    dtype DataType.FLOAT
```

But `CHW4` and `HWC8` are defined for INT8 and FP16, and asking for one on a float
tensor is rejected at build time:

```
format CHW4   : PolygraphyException -- has dataType Float unsupported by tensor's allowed TensorFormats
format HWC8   : PolygraphyException
```

The usual fix — change the tensor's dtype — is exactly what `SetTensorDatatypes`
can no longer do. So on a parsed ONNX, the reachable set of formats is decided by
the ONNX file. Reaching the rest means building the network by hand with the input
declared INT8 or FP16.

## The general escape hatch

`PostprocessNetwork(network, func)` takes an arbitrary function over the
`INetworkDefinition` and stays lazy. The three loaders above are conveniences over
it; when the dedicated one is gone, this is the way in.

Marking an intermediate tensor as an extra output, which none of them can do:

```
marked `relu` from layer `node_relu`
engine outputs: ['y', 'z', 'relu']
```
