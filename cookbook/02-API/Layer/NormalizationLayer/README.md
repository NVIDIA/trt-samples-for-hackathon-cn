# Normalization Layer

+ Computation process:
$$
T3 = \frac{T0 - \bold{mean}\left(T0,axes\right)}{\sqrt{\bold{var}\left(T0,axes\right)+\epsilon}} \times T1 + T2
$$

## Case Layer Normalization

+ Shape of T0, T3: [$d_0, d_1, d_2, d_3, \dots$]
+ Shape of T1, T2: [$a_0, a_1, a_2, a_3, \dots$], $a_i == d_i$ if 1 << i is set as normalization axes, or ai == 1 if not set.

# Case Group Normalization

+ Shape of T0, T3: [N, C, H, W, ...]
+ Shape of T1, T2: [1, G, 1, 1, ...], G is the number of groups, s.t. C % G == 0

## Case Instance Normalization
+ Shape of T0, T3: [N, C, H, W, ...]
+ Shape of T1, T2: [1, C, 1, 1, ...]
