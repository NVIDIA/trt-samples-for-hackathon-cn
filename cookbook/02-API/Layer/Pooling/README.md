# Pooling layer

+ Pooling layer.

+ Steps to run.

```bash
python3 main.py
```

+ Attributes

| Name | Description |
| :---------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: |
| pooling_type | Specifies the pooling operation: MAX, AVERAGE, or MAX_AVERAGE_BLEND |
| blend_factor | Blending factor used when pooling_type is MAX_AVERAGE_BLEND. Default: 0.5 |
| padding_mode | Determines padding calculation: EXPLICIT_ROUND_DOWN, EXPLICIT_ROUND_UP, SAME_UPPER, SAME_LOWER. Default: EXPLICIT_ROUND_DOWN |
| padding_nd | Symmetric padding applied to each spatial dimension. Default: (0, 0) |
| stride_nd | Stride along each spatial dimension. Default: same as window_size_nd |
| pre_padding | Padding at the start of each spatial dimension. Default: (0, 0) |
| post_padding | Padding at the end of each spatial dimension. Default: (0, 0) |
| average_count_excludes_padding | If True, average pooling ignores padded input elements. Default: False |

+ Available values of `trt.PoolingType`.

|       Name        |                                      Comment                                       |
| :---------------: | :-------------------------------------------------------------------------------: |
|      AVERAGE      |             Output the average of the values in the sampling window.              |
|        MAX        |             Output the maximum of the values in the sampling window.              |
| MAX_AVERAGE_BLEND | $output = \left(1 - blendFactor\right) \times MAX + blendFactor \times AVERAGE$  |

+ Available values of `trt.PaddingMode`. Output length of each spatial dimension is computed from the padded input length $M$, dilated kernel size $DK$ and stride $S$.

|        Name         |                                        Comment                                         |
| :-----------------: | :-----------------------------------------------------------------------------------: |
| EXPLICIT_ROUND_DOWN | Use explicit padding, $output = \lfloor \frac{M - DK}{S} \rfloor + 1$ (round down).    |
|  EXPLICIT_ROUND_UP  | Use explicit padding, $output = \lceil \frac{M - DK}{S} \rceil + 1$ (round up).        |
|     SAME_UPPER      | Use SAME padding to keep output size $\lceil \frac{input}{S} \rceil$, with pre-padding $\le$ post-padding. |
|     SAME_LOWER      | Use SAME padding to keep output size $\lceil \frac{input}{S} \rceil$, with pre-padding $\ge$ post-padding. |

+ Priority of padding APIs: padding_mode > pre_padding = post_padding > padding_nd

+ Input / output data type: T in [float16, float32, bfloat16, int8, float8].

+ Shape: input rank $\ge 3$; output rank equals input rank. Volume $\le 2^{31}$ ($\le 2^{40}$ if all shapes are build-time constants).
