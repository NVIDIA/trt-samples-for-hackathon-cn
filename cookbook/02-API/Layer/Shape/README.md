# Shape layer

+ Shape layer. Outputs the shape (dimensions) of the input tensor as a shape tensor. See `case_simple` in `main.py`.

+ Steps to run.

```bash
python3 main.py
```

+ Input / output data types.

| Item   | Data Type                                                                              |
| :----- | :------------------------------------------------------------------------------------- |
| input  | `bool`, `int4`, `int8`, `int32`, `int64`, `float8`, `float16`, `float32`, `bfloat16`   |
| output | `int64` (a 1-D shape tensor)                                                           |

+ Shape constraints.

  + For an input of shape `[a0, ..., an]` (`n >= 0`), the output is a 1-D shape tensor of length `rank(input)` holding the values `[a0, ..., an]`.
  + When the input is a scalar (`n == 0`), the output is an empty tensor.
