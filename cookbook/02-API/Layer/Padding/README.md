# Padding layer

+ Padding layer.

+ Steps to run.

```bash
python3 main.py
```

+ Pad zeros around (or crop) the last two dimensions of the input tensor, refer to `case_simple` for padding and `case_crop` for cropping.

+ IPaddingLayer is deprecated since TensorRT 8.2; use the Slice layer instead, which additionally supports non-constant padding, reflect / clamp padding modes and dynamic output shape.

+ Only padding or cropping on the last two dimensions are supported.

+ Attributes.

|      Name       |                                       Description                                        |       Default        |
| :-------------: | :-------------------------------------------------------------------------------------: | :------------------: |
| pre_padding_nd  | Amount of pre-padding for the last two dimensions. Positive pads with zeros; negative trims. | set by constructor |
| post_padding_nd | Amount of post-padding for the last two dimensions. Positive pads with zeros; negative trims. | set by constructor |

+ Input / output data type: T in [int8, int32, float16, float32]; input and output share dtype T.

+ Shape: input rank $\ge 4$. Output shape [b₀,...,bₙ₋₁]:
  + For dimensions before the last two: $b_i = a_i$.
  + For the two innermost dimensions: $b_i = a_i + pre_i + post_i$.
