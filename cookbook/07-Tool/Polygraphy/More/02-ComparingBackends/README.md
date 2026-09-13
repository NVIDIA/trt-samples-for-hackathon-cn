# Comparing backends

+ Run one model through TensorRT and onnxruntime, then pick a comparison that means something.

+ Steps to run.

```bash
python3 main.py
```

## The number everything is measured against

```
output y: float32, max |trt - ort| = 7.451e-09
output z: int64,   identical = True
```

Ordinary FP32 rounding. The upstream example compares an *identity* model where
the two backends agree exactly, so the comparison never fails and the reader
never sees what any of the alternatives are for. Here every comparison below runs
against that same 7.45e-09.

## `simple` is a threshold somebody has to choose

```
simple(atol=1e-8)  above the real diff  : PASS
simple(atol=1e-9)  below the real diff  : FAIL
```

One order of magnitude flips the verdict. A `simple` check that passes says
nothing until you know which number it used — and the default is a guess, not a
correctness criterion.

## The metric-based alternatives judge the whole tensor

```
distance_metrics(l2=1e-5,  cos=0.99)   : PASS      L2 = 9.80e-09, cosine = 1
distance_metrics(l2=1e-12, cos=0.99)   : FAIL
quality_metrics(psnr=50,  snr=25)      : PASS      PSNR = 144.69 dB, SNR = 140.52 dB
quality_metrics(psnr=300, snr=25)      : FAIL
perceptual_metrics()                   : needs `pip install lpips`
```

They are **not automatically more lenient** — tightened past the real difference
they fail too. What they buy is a judgement about the tensor as a whole rather
than about its single worst element.

Note on `perceptual_metrics`: without `lpips` installed Polygraphy logs the
missing import and **still returns a verdict**. That verdict means nothing, so
the case skips it rather than printing a PASS that looks real.

## The trap: `indices` needs indices

```
indices() straight on the raw outputs   : FAIL
indices() after PostprocessFunc.top_k   : PASS
```

`CompareFunc.indices` compares results *containing indices*. Applied to raw
logits it treats each float as an index value and fails —

```
[E] FAILED | Value: 0.012981239706277847 not found in output
```

— even though the model's own `argmax` output is bit-identical between the two
backends. The intended pairing is `PostprocessFunc.top_k` first:

```python
top_k = Comparator.postprocess(run_both(), PostprocessFunc.top_k(k={"y": (5, 1)}))
Comparator.compare_accuracy(top_k, compare_func=CompareFunc.indices(index_tolerance=0))
```

That answers "do the backends rank the classes the same way", which is usually
the real question for a classifier and is immune to the rounding `simple` is so
sensitive to. Note `k` is scoped to `y` by name — `z` is 1-D and `top_k` on
axis 1 of a 1-D array raises `AxisError`.

## `CreateConfig(fp16=True)` is refused, not ignored

```
PolygraphyException: fp16 in CreateConfig is not available on TensorRT version 11.1.0.106.
```

Networks parsed from ONNX come out **strongly typed**, so the builder has no
precision to choose and the flag is meaningless.

Worth contrasting with Torch-TensorRT, where the equivalent
`enabled_precisions={torch.float16}` is accepted, ignored and never warned about
(see `06-DLFrameworkTRT/Torch-TensorRT/MixedPrecisionAutocast/`). Same underlying
TensorRT 11 change, opposite ergonomics — **loud refusal beats silent no-op**.

## Related

+ [`../01-LazyVsImmediate/`](../01-LazyVsImmediate/README.md) — why the runners here take lazy loaders.
