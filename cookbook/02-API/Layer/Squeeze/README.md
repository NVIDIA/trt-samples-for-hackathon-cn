# Squeeze layer

+ Squeeze layer. Reshapes the input tensor by removing the dimensions specified by `axes`; each such dimension must have length 1. See `case_simple` in `main.py`.

+ Steps to run.

```bash
python3 main.py
```

+ Input / output data types.

| Item          | Data Type                                                                            |
| :------------ | :---------------------------------------------------------------------------------- |
| input0 (data) | `bool`, `int4`, `int8`, `int32`, `int64`, `float8`, `float16`, `float32`, `bfloat16` |
| input1 (axes) | `int32`, `int64`                                                                    |
| output        | same type `T` as input0                                                             |

+ Shape constraints.

  + `input1` is a 1-D tensor of shape `[n]` listing the axes to remove; each listed dimension of `input0` must have length 1.
  + `rank(output) == rank(input0) - n`.
