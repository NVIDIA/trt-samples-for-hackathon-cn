# TopK layer

+ TopK layer.

+ Steps to run.

```bash
python3 main.py
python3 benchmark_plugin.py   # ITopKLayer vs the TopkLastDim plugin
```

+ Find the `k` largest (or smallest) elements along one axis of the input tensor, returning both the values and their indices. Refer to `case_simple` for the basic usage, `case_deprecated` for the 3-argument constructor, and `case_shape_input` / `case_dds` for feeding `k` from another tensor at runtime.

+ Available values of `trt.TopKOperation`.

| Name |                     Comment                      |
| :--: | :----------------------------------------------: |
| MAX  |  Select the `k` largest elements along the axis.  |
| MIN  | Select the `k` smallest elements along the axis. |

+ `trt.ReduceOperation` is also listed at the end of `main.py` because `case_dds` uses a Reduce layer to compute `k` at runtime; see the `Reduce` layer for its meaning.

+ Attributes.

|     Name     | Description                                                                           | Default |
| :----------: | :----------------------------------------------------------------------------------- | :-----: |
| op           | TopK operation, one of `trt.TopKOperation`.                                           |    -    |
| k            | Number of elements to keep. `k <= d` (the axis length) and `k <= 3840`.               |    -    |
| axes         | Bitmask selecting the single reduced axis (must be one of the last four dimensions).  |    -    |
| indices_type | Data type of the output indices tensor, `trt.DataType.INT32` or `trt.DataType.INT64`. |  INT32  |

+ Input / output data-type and shape constraints:
  + Input tensor `T1` in [int32, int64, float16, float32, bfloat16]; output values share type `T1`, output indices `T2` in [int32, int64].
  + For input shape `[a0,...,an]`, both outputs have shape `[b0,...,bn]` where `bi == ai` except on the reduced axis `i == log2(axes)`, where `bi == k`.
  + When two elements share the same value, the one with the smaller index is selected.

## `ITopKLayer` vs the `TopkLastDim` plugin

TensorRT 11.1 ships a **`TopkLastDim`** plugin inside its own plugin library (registered by
`init_libnvinfer_plugins`, nothing to build or load), built on the AIR — Adaptive Iterative
Radix — sort kernel from TensorRT-LLM. It covers the same operation, so `benchmark_plugin.py`
measures when it is worth reaching for. Both answers are checked against NumPy, so a faster
wrong answer cannot win.

Measured on B200 (idle), TensorRT 11.1.0.106, float32 `[256, 16384]`, top-k along the last axis,
median of 50 iterations:

| k | `ITopKLayer` | `TopkLastDim` plugin | speed-up |
| ---: | ---: | ---: | ---: |
| 8 | 0.039 ms | 0.047 ms | **0.83x** (the layer wins) |
| 64 | 0.109 ms | 0.051 ms | **2.12x** |
| 1024 | 0.130 ms | 0.089 ms | **1.47x** |

So it is not a free upgrade: at small k the plugin is *slower*, and the crossover is somewhere
between k=8 and k=64 on this shape. The layer's cost grows quickly with k (0.039 → 0.130 ms) while
the plugin's barely moves (0.047 → 0.089 ms), which is the radix-sort behaviour you would expect.

**The hard limit decides it for large k.** `ITopKLayer` accepts k ≤ 3840 and the plugin has no such
ceiling:

```txt
ITopKLayer with k=3840: built
ITopKLayer with k=3841: REFUSED (network.add_topk returned None)
```

Note *how* it refuses: `network.add_topk` returns `None` immediately. The builder is never reached,
so there is no error message to read and no `parser.get_error` to inspect — an unchecked return
value here becomes an `AttributeError` on `NoneType` several lines later.

**Indices are `int32` from both**, in this configuration. The ONNX `TopK` specification says
`int64`, so a graph parsed from ONNX needs a `Cast` either way; the plugin's README lists this as a
known deviation, but it is not a difference between the two implementations.
