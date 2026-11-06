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
against that same measured difference. The example measures it rather than quoting
it, because it moves with the TensorRT build: it was 7.45e-09 on 11.0 and is
3.725e-09 on 11.3.

## `simple` is a threshold somebody has to choose

```
simple(atol=1e-8)            default rtol : PASS
simple(atol=1e-9)            default rtol : PASS
simple(atol=1e-8, rtol=0)  above real diff: PASS
simple(atol=1e-9, rtol=0)  below real diff: FAIL
```

One order of magnitude flips the verdict — but only once `rtol` is out of the way.
The test is `|a - b| <= atol + rtol * |b|`, so with the default `rtol` still in
place the first two **both pass**: for outputs of order 0.06 the `rtol` term is
~6e-7, which swamps either `atol`. Tightening `atol` alone did nothing at all.

A `simple` check that passes says nothing until you know *both* numbers it used —
and the defaults are a guess, not a correctness criterion.

## The metric-based alternatives judge the whole tensor

```
L2(1e-5)   above the real diff          : PASS      L2 = 6.11e-09
L2(1e-12)  below the real diff          : FAIL
CosineSimilarity(0.99)                  : PASS
Psnr(50)                                : PASS      PSNR = 146.5 dB
Psnr(300)  unreachably strict           : FAIL
Snr(25)                                 : PASS
PerceptualMetrics()                     : needs `pip install lpips`
```

They are **not automatically more lenient** — tightened past the real difference
they fail too. What they buy is a judgement about the tensor as a whole rather
than about its single worst element.

### One metric per compare func (changed in polygraphy 0.5x)

This is where code written against polygraphy 0.49 stops working. Two combined
factory methods were split into one class per metric:

| polygraphy 0.49 | now |
| --- | --- |
| `CompareFunc.distance_metrics(l2_tolerance=…, cosine_similarity_threshold=…)` | `L2CompareFunc(l2_threshold=…)` + `CosineSimilarityCompareFunc(cosine_similarity_threshold=…)` |
| `CompareFunc.quality_metrics(psnr_tolerance=…, snr_tolerance=…)` | `PsnrCompareFunc(psnr_threshold=…)` + `SnrCompareFunc(snr_threshold=…)` |
| `CompareFunc.perceptual_metrics(…)` | `PerceptualMetricsCompareFunc(…)` (the old name still exists, deprecated) |

`CompareFunc` itself keeps only `simple`, `indices` and the deprecated
`perceptual_metrics`. The split costs one `compare_accuracy` call per metric and
buys an explicit verdict per metric — a combined call could only report one
verdict for two independent questions.

Note on `PerceptualMetricsCompareFunc`: without `lpips` installed Polygraphy logs
the missing import and **still returns a verdict**. That verdict means nothing, so
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
