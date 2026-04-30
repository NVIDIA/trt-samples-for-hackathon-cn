# Normalization layer

+ Normalization layer.

+ Steps to run.

```bash
python3 main.py
```

+ Normalize the input tensor with scale (T1) and bias (T2), refer to `case_layer_normalization`, `case_group_normalization` and `case_instance_normalization`.

+ Computation process:

$$
T3 = \frac{T0 - \bold{mean}\left(T0,axes\right)}{\sqrt{\bold{var}\left(T0,axes\right)+\epsilon}} \times T1 + T2
$$

+ Attributes.

|     Attribute     |                                              Description                                              |     Default      |    Range    |
| :---------------: | :------------------------------------------------------------------------------------------------: | :--------------: | :---------: |
| axes              | Bitmask of the axes to normalize on (bit $i$ set means axis $i$ is normalized). Set by constructor. | set by constructor | bitmask   |
| epsilon           | Small value added to the variance to avoid division by 0.                                          | $10^{-5}$        | > 0         |
| num_groups        | Number of groups; if $\ne 1$, input channels are split into groups before normalization.           | 1                | divides C   |
| compute_precision | Data type in which the normalization computation is performed.                                     | DataType.FLOAT   | float32 / float16 |

+ Input / output data type: T in [float32, float16, bfloat16]; scale (T1) and bias (T2) share dtype T.

+ Layer Normalization

+ Shape of T0, T3: [$d_0, d_1, d_2, d_3, \dots$]
+ Shape of T1, T2: [$a_0, a_1, a_2, a_3, \dots$], $a_i == d_i$ if 1 << i is set as normalization axes, or $a_i == 1$ if not set.

+ Group Normalization

+ Shape of T0, T3: [N, C, H, W, ...]
+ Shape of T1, T2: [1, C, 1, 1, ...], with num_groups groups G s.t. C % G == 0

+ Instance Normalization

+ Shape of T0, T3: [N, C, H, W, ...]
+ Shape of T1, T2: [1, C, 1, 1, ...]
