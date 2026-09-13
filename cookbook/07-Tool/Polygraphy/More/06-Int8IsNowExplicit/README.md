# INT8 without a calibrator

+ What replaced `polygraphy.backend.trt.Calibrator` on TensorRT 11.

+ Steps to run.

```bash
python3 main.py
```

## The old flow is gone, and it fails late

Polygraphy 0.50.3 still ships a `Calibrator` that wraps TensorRT's
implicit-quantization flow. On TensorRT 11 the API it wraps no longer exists:

```
trt.BuilderFlag.INT8         : REMOVED
trt.IInt8Calibrator          : REMOVED
trt.IInt8EntropyCalibrator2  : REMOVED
trt.IInt8MinMaxCalibrator    : REMOVED

CreateConfig(int8=True)      : PolygraphyException: int8 in CreateConfig is not available on TensorRT version 11.1...
CreateConfig(calibrator=...) : AttributeError: module 'tensorrt' has no attribute 'IInt8EntropyCalibrator2'
```

Note the second line: `from polygraphy.backend.trt import Calibrator` **succeeds**,
and so does constructing one with a data loader. The failure only arrives inside
TensorRT when the engine is built. Code written against the upstream
`04_int8_calibration_in_tensorrt` example looks fine until it runs.

## Quantization lives in the graph now

```
float model:  12 nodes, QuantizeLinear 0, DequantizeLinear 0
QAT model  :  28 nodes, QuantizeLinear 8, DequantizeLinear 8
```

`QuantizeLinear` / `DequantizeLinear` pairs carry the scales as initializers, put
there by a quantization toolkit (NVIDIA ModelOpt, `pytorch-quantization`, or QAT).
**Nothing is passed to the builder** — the difference between the two engines is
entirely a property of the two ONNX files.

This is the same move FP16 made: from a builder flag to a declaration in the
graph. See [`../05-BuildNetworkByHand/`](../05-BuildNetworkByHand/README.md).

## Comparing a quantized engine

```
max |float - qat| on logits : 9.9610   <- INT8 is supposed to differ
same predicted class        : True     <- the question that actually matters
```

`CompareFunc.simple` with any sane tolerance fails here, and it should: the
quantized model computes something different on purpose. For a classifier the
useful question is whether the ranking survives, which is what
[`../02-ComparingBackends/`](../02-ComparingBackends/README.md) reaches for
`PostprocessFunc.top_k` + `CompareFunc.indices` to answer.

## Migration summary

| TensorRT 10 and earlier | TensorRT 11 |
| --- | --- |
| `CreateConfig(int8=True, calibrator=Calibrator(...))` | quantize the ONNX offline, build normally |
| calibration cache managed at build time | scales are initializers in the model |
| implicit quantization, builder chooses | explicit quantization, graph decides |

## Related

+ [`../05-BuildNetworkByHand/`](../05-BuildNetworkByHand/README.md) — the same move for FP16.
+ [`../04-ExtendInterop/`](../04-ExtendInterop/README.md) — the full list of surviving `BuilderFlag`s.
+ Cookbook skill `trt-strong-typing-migration`.
