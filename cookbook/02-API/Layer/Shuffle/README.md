# Shuffle layer

+ Shuffle layer.

+ Steps to run.

```bash
python3 main.py
```

## Attributes

| Attribute | Description | Default |
| --- | --- | --- |
| `first_transpose` | The permutation applied by the first transpose operation. | Identity Permutation |
| `reshape_dims` | The reshaped dimensions. The product of the dimensions must be equal to the product of the input dimensions. Special values: `0` copies the corresponding input dimension; `-1` infers dimension from input and other reshape dimensions (only one permitted). | N/A |
| `second_transpose` | The permutation applied by the second transpose operation. | Identity Permutation |
| `zero_is_placeholder` | The meaning of 0 in the reshape dimensions. If `true`, a 0 denotes copying the corresponding dimension from the first input tensor. If `false`, a 0 represents a zero-length dimension. | `true` |

+ Input / output data types and shapes:
  + Input tensor (`T`) and output tensor share the same data type: `T` in [bool, int4, int8, uint8, int32, float8, float16, float32, bfloat16].
  + The optional second input (reshape dimensions, set via `set_input(1, ...)`) is an int32 or int64 tensor of shape `[n]`; the output then has rank `n`.
  + The transposes permute axes; the reshape only changes the shape, so the total volume is preserved.
