# Scale layer

+ Scale layer.

+ Steps to run.

```bash
python3 main.py
```

+ Apply a per-element affine-then-power transform to the input tensor, refer to `case_simple` (UNIFORM), `case_channel` (CHANNEL), `case_element` (ELEMENTWISE) and `case_scale_channel_axis` (custom channel axis).

+ Computation process:

$$
output = \left(input \times scale + shift\right)^{power}
$$

+ Available values of `trt.ScaleMode`.

|    Name     |                                   Comment                                   |
| :---------: | :------------------------------------------------------------------------: |
| UNIFORM     | Identical coefficients for all elements (each of scale/shift/power holds 1 value). |
| CHANNEL     | Identical coefficients for all elements in the same channel (one value per channel). |
| ELEMENTWISE | Identical coefficients for all elements with the same channel and spatial coordinates. |

+ Attributes.

|   Attribute  |                                     Description                                     |    Default    |
| :----------: | :-------------------------------------------------------------------------------: | :-----------: |
| mode         | Scale mode, one of `trt.ScaleMode`. Set by constructor.                          | set by constructor |
| scale        | Scale coefficient tensor.                                                        | 1 (if empty)  |
| shift        | Shift coefficient tensor.                                                        | 0 (if empty)  |
| power        | Power coefficient tensor.                                                         | 1 (if empty)  |
| channel_axis | Axis treated as the channel dimension (used by CHANNEL / ELEMENTWISE modes). `add_scale` sets it to 1; `add_scale_nd` sets it from its argument. | 1 (via `add_scale`) |

+ Input / output data type: T in [int8, float16, float32, bfloat16]; weights share dtype T; input and output share the same shape.

+ Shape: input rank $\ge 4$.
