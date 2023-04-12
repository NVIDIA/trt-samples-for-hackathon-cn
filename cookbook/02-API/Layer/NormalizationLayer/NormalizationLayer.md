# Normalization Layer

+ Common
+ Simple example

---

## Common

+ Algorithm
$$
Y = \frac{X - \bold{Mean}\left(X,axes\right)}{\sqrt{\bold{Var}\left(X,axes\right)+\epsilon}} \times S + B
$$

+ Input tensor
  + T0
  + T1, scale
  + T2, bias

+ Output tensor
  + T3

+ Data type
  + T0, T1, T2, T3: float32, float16

+ Shape
  + Instance Normalization
    + T0, T3 [N, C, H, W, ...]
    + T1, T2 [1, C, 1, 1, ...]

  + Group Normalization
    + T0, T3 [N, C, H, W, ...]
    + T1, T2 [1, G, 1, 1, ...], G is the number of groups

  + Layer Normalization
    + T0, T3 [$d_0, d_1, d_2, d_3, \dots$]
    + T1, T2 [$a_0, a_1, a_2, a_3, \dots$], $a_i == d_1$ if 1 << i is set as normalization axes, or ai == 1 if not set.

+ Attribution and default value
  + epsilon, value in denominator to avoid division by 0.
  + axes, axes mask to perform normalization.
  + num_groups = 1, the number of groups of the normalization.
  + compute_precision, the precision of the computation (especially in accumulation).
    + By experience, using FP16 precision in normalization layers during building a model with large size of normalization dimensions (>=4096 for example) in FP16 mode causes relatively big error, so we can setback to FP32 precision to reduce error as cost of a bit of performance.

---

## Instance Normalization

+ Refer to InstanceNormalization.py

+ Do a instance normalization.

---

## Group Normalization

+ Refer to GroupNormalization.py

+ Do a group normalization.

---

## Layer Normalization

+ Refer to LayerNormalization.py

+ Do a layer normalization.
