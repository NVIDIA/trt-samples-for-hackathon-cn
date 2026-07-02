# TopK layer

+ TopK layer.

+ Steps to run.

```bash
python3 main.py
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
