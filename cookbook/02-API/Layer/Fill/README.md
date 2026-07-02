# Fill layer

+ Fill layer.

+ Steps to run.

```bash
python3 main.py
```

+ Generate a tensor of a given shape filled either with evenly spaced numbers or with random values. Refer to `case_linspace_1` / `case_linspace_2` for `LINSPACE`, `case_random_normal` for `RANDOM_NORMAL`, `case_random_uniform` for `RANDOM_UNIFORM`, and `case_shape_input` / `case_dds` for feeding the output shape at runtime.

+ Available values of `trt.FillOperation`.

|      Name      |                                        Comment                                        |
| :------------: | :-----------------------------------------------------------------------------------: |
|    LINSPACE    | Evenly spaced numbers; $\alpha$ is the start value, $\beta$ is the per-dimension step. |
| RANDOM_UNIFORM |   Values drawn from a uniform distribution; $\alpha$ is the min, $\beta$ is the max.   |
| RANDOM_NORMAL  |  Values drawn from a normal distribution; $\alpha$ is the mean, $\beta$ is the std-dev.  |

+ Attributes.

| Attribute  | Description                                                                 | Default | Range |
| :--------: | :------------------------------------------------------------------------- | :-----: | :---: |
| dimensions | Shape of the output tensor.                                                | -       | -     |
| operation  | Fill behavior, see `trt.FillOperation`.                                    | -       | -     |
| alpha      | Start (LINSPACE) / mean (RANDOM_NORMAL) / min (RANDOM_UNIFORM).            | 0       | -     |
| beta       | Step (LINSPACE) / std-dev (RANDOM_NORMAL) / max (RANDOM_UNIFORM).          | 1       | -     |
| to_type    | Data type of the output tensor.                                            | float32 | -     |

+ Input / output data types and shapes:
  + Inputs are all optional: input0 (int32/int64, shape `[n]`) sets the output dimensions; input1 (scalar) sets `alpha`; input2 sets `beta` (shape `[n]` for LINSPACE, scalar otherwise).
  + Output data type `T`: LINSPACE supports int32, int64, float32; RANDOM_UNIFORM / RANDOM_NORMAL support float16, float32.
  + Output shape is `[a0,...,an]` from the `dimensions` attribute (or from input0).

+ Random seed is not supported in TensorRT yet.
