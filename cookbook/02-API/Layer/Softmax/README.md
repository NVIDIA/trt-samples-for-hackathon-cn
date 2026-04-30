# SoftMax layer

+ SoftMax layer.

+ Steps to run.

```bash
python3 main.py
```

+ Normalize the input tensor along a single axis so that the values fall in $[0, 1]$ and sum to 1 per slice: $output_i = \frac{\exp\left(x_i\right)}{\sum_j \exp\left(x_j\right)}$. Refer to `case_simple` for normalizing along the second axis.

+ Only one axis can be set at a time; combining bits such as `axes=(1<<2)+(1<<3)` is not allowed.

+ Attributes

| Name |                              Description                              |          Default          |
| :--: | :-----------------------------------------------------------------: | :-----------------------: |
| axes | Bitmask selecting the single axis to normalize (must set exactly one bit). | `1 << max(0, Rank - 3)` |

+ Input / output data-type and shape constraints:
  + Input and output share the same data type `T` in [float16, float32, bfloat16].
  + Input and output have the same shape `[a0, ..., an]`.
