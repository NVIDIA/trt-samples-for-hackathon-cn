# ONNX PTQ Method

+ Calibration **method** on an ONNX model — entropy vs max, per-node calibration, and INT4 weight-only.

+ Steps to run.

```bash
python3 main.py
```

[`../../03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/`](../../03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/README.md)
covers FP8 PTQ with max calibration. The choice it does not make visible is the **calibration
method**, which decides where the clipping threshold lands:

| method | threshold | effect |
| --- | --- | --- |
| `max` | the largest value seen | nothing clipped; outliers get a scale that spends resolution on values almost nothing uses |
| `entropy` | minimises information loss vs the FP32 distribution | outliers **are** clipped; the bulk of the distribution gets more levels |

Measured on B200, TensorRT 11.1.0.106, `nvidia-modelopt` 0.44.0, MNIST CNN, INT8, with two large
outliers planted in the calibration set on purpose.

## Neither method wins — and measuring one probe would have said otherwise

```txt
on OUTLIER-bearing input : entropy 6.82%  vs  max 6.12%   -> max wins
on TYPICAL input         : entropy 5.76%  vs  max 6.61%   -> entropy wins
```

That is the whole trade in two rows. `max` sizes its scale to represent the extremes, so it
necessarily wins on inputs that contain them — and pays for it everywhere else. `entropy` clips the
tail so the bulk of the distribution gets more levels.

**This example originally measured only the outlier-bearing probe**, which is the same data the
outliers were planted in. That is a rigged comparison: `max` wins there by construction. It produced
a confident recommendation that reversed as soon as a second, ordinary probe was added. If you are
choosing a calibration method, evaluate on the inputs you actually serve — and on more than one kind.

## Per-node calibration is a memory strategy, and it is not free

`calibrate_per_node=True` walks the graph instead of collecting activations for the whole model at
once, which is how a model too large to calibrate in one pass gets calibrated at all.

```txt
calibrate_per_node=False :   0.7 s, 6.82%
calibrate_per_node=True  :  35.0 s, 6.85%
bit-identical: False   accuracy drift: 0.03%   time ratio: 42–49x (run to run)
```

**~45x slower**, and the accuracy is *not* bitwise identical. The 0.03% drift is expected rather
than alarming: collecting per node changes the order in which histograms accumulate, so a chosen
threshold can land a bin away. Treat "unchanged" as **within a bin**, not as bitwise equal — and do
not reach for this flag unless memory is the reason.

## INT4 weight-only: quantizes, then fails to build

```txt
int4 / awq_clip: quantized in 1.2 s
TensorRT: Assertion failed: inputSize % scaleSize == 0: Inferred block size is not an integer.
          Input volume = 3211264, scale volume = 25600
```

ModelOpt produces the file happily; the TensorRT parser rejects it because the AWQ block size does
not divide this Gemm's weight volume evenly. Block-quantized weights only work when the shapes line
up — the same constraint [`../../02-API/Layer/QDQStructure/`](../../02-API/Layer/QDQStructure/README.md)
shows from the network-API side, where the scale tensor's shape is *derived* from `block_shape`
rather than chosen.

## A build error that names the wrong thing

The INT8 engines initially reported `build failed`. The real message:

```txt
Network has dynamic or shape inputs, but no optimization profile has been defined
```

The model has a dynamic batch dimension, and the error mentions neither quantization nor the input
name — so on a quantization example it reads like a quantization problem. It is not. `build_engine`
here derives a profile from whatever inputs are dynamic.

## Related

+ [`../LowBitQuantization/`](../LowBitQuantization/README.md) — the same comparison from PyTorch, and
  which low-bit formats actually reach an engine.
+ [`../../02-API/Layer/QDQStructure/`](../../02-API/Layer/QDQStructure/README.md) — Q/DQ granularity
  at the network level, including block quantization.
+ [`../../05-Plugin/ONNXPTQWithPlugin/`](../../05-Plugin/ONNXPTQWithPlugin/README.md) — PTQ when the
  graph contains an op ONNX Runtime cannot execute.
