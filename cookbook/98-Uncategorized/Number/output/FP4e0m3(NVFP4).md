# FP4e0m3(NVFP4) - not IEEE754

+ Sign:     $s$ ($p=1$ bit)
+ Exponent: $e$ ($q=0$ bits)
+ Mantissa: $m$ ($r=3$ bits)

+ Normal value ($1_2 \le e_2 \le 0_2$) (subscript 2 represents the base)

$$
\begin{equation}
\begin{aligned}
E &= e - \left( 2^{q-1} - 1 \right) = e - 0 \\
M &= m \cdot 2^{-r} = m \cdot 2^{-3} \\
value &= \left( -1 \right) ^ {s} 2 ^ {E} \left( 1 + M \right) = \left( -1 \right) ^ {s} 2 ^ {-2} 2 ^ {e} \left( m  + 2^{3} \right)
\end{aligned}
\end{equation}
$$

+ Subnormal value ($e_2 = _2$)

$$
\begin{equation}
\begin{aligned}
E &= 2 - 2^{q-1} = 1 \\
M &= m \cdot 2^{-r} = m \cdot 2^{-3} \\
value &= \left( -1 \right) ^ {s} 2 ^ {E} M = \left( -1 \right) ^ {s} 2^{1} m
\end{aligned}
\end{equation}
$$

+ Special value

| Exponent  | Mantissa  | meaning         |
| :-:       | :-:       | :-:             |
| all 0     | all 0     | Signed Zero     |
| all 0     | not all 0 | Subnormal Value |
| all 1 | all 0     | Normal value    |
| all 1 | not all 0 | Normal value    |

+ Examples

| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponent}\color{#0000FF}{Mantissa}$) | value          | comment          |
| :-:        | :-:            | :-:              |
| $\color{#FF0000}{0}\color{#007F00}{}\color{#0000FF}{000}$ | $+0$               | Positive Zero    |
| $\color{#FF0000}{0}\color{#007F00}{}\color{#0000FF}{001}$ | $1$                |                  |
| $\color{#FF0000}{0}\color{#007F00}{}\color{#0000FF}{111}$ | $2.4749\times10^{+00}$ | Maximum |
| $\color{#FF0000}{1}\color{#007F00}{}\color{#0000FF}{111}$ | $-2.4749\times10^{+00}$ | Maximum negative |
| $\color{#FF0000}{1}\color{#007F00}{}\color{#0000FF}{000}$ | $-0$               | Positive Zero    |

+ All non-negative values

| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponen}\color{#0000FF}{Mantissa}$) | value |
| :-:        | :-:   |
| $\color{#FF0000}{0}\color{#007F00}{}\color{#0000FF}{000}$     | 0.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{}\color{#0000FF}{001}$     | 3.536e-01  |
| $\color{#FF0000}{0}\color{#007F00}{}\color{#0000FF}{010}$     | 7.071e-01  |
| $\color{#FF0000}{0}\color{#007F00}{}\color{#0000FF}{011}$     | 1.061e+00  |
| $\color{#FF0000}{0}\color{#007F00}{}\color{#0000FF}{100}$     | 1.414e+00  |
| $\color{#FF0000}{0}\color{#007F00}{}\color{#0000FF}{101}$     | 1.768e+00  |
| $\color{#FF0000}{0}\color{#007F00}{}\color{#0000FF}{110}$     | 2.121e+00  |
| $\color{#FF0000}{0}\color{#007F00}{}\color{#0000FF}{111}$     | 2.475e+00  |
