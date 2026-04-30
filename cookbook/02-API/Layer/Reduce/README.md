# Reduce layer

+ Reduce layer.

+ Steps to run.

```bash
python3 main.py
```

+ Reduce the input tensor along one or more axes selected by a bitmask. Refer to `case_simple` for a sum-reduce on the second axis.

+ More than one axis can be reduced at the same time by combining bits, for example `axes=(1<<2)+(1<<3)` reduces dimensions 2 and 3.

+ Available values of `trt.ReduceOperation` (`x_i` are the elements along the reduced axes).

| Name |                         Comment                          |
| :--: | :------------------------------------------------------: |
| SUM  |          $output = \sum_{i} x_i$ (sums elements)          |
| PROD |         $output = \prod_{i} x_i$ (multiplies elements)         |
| MAX  |         $output = \max_{i} x_i$ (maximum element)          |
| MIN  |         $output = \min_{i} x_i$ (minimum element)          |
| AVG  | $output = \frac{1}{N} \sum_{i} x_i$ (average of $N$ elements) |

+ Attributes

|   Name    |                                          Description                                          | Default |
| :-------: | :-----------------------------------------------------------------------------------------: | :-----: |
|    op     |                     Reduce operation, one of `trt.ReduceOperation`.                       |   SUM   |
|   axes    | Bitmask of the axes to reduce (e.g. `axes=6` reduces dimensions 1 and 2). Set by constructor. |    -    |
| keep_dims |  If True, keep the reduced dimensions as size 1; if False, remove them (reducing the rank).  |  False  |

+ Input / output data-type and shape constraints:
  + Input and output share the same data type `T` in [int8, int32, int64, float16, float32, bfloat16].
  + Input rank must be >= 1.
  + With `keep_dims=True` the output has the same rank as the input with reduced axes set to length 1; with `keep_dims=False` the reduced axes are removed.
