# Select layer

+ Select layer. Selects elements from `thenInput` where `condition` is true, otherwise from `elseInput`. See `case_simple` in `main.py`.

+ Steps to run.

```bash
python3 main.py
```

+ Input / output data types.

| Item            | Data Type                                                    |
| :-------------- | :---------------------------------------------------------- |
| condition       | `bool`                                                      |
| thenInput       | `T`: `int32`, `int64`, `float16`, `float32`, `bfloat16`, `bool` |
| elseInput       | `T` (same as thenInput)                                     |
| output          | `T` (same as thenInput)                                     |

+ Shape / broadcast constraints.

  + `condition`, `thenInput` and `elseInput` must have the same rank.
  + For each dimension the lengths must either match or one must be 1 (in which case that tensor is broadcast along that axis).
  + The `output` has the same rank as the inputs; each output dimension takes the non-1 length among the corresponding input dimensions.
