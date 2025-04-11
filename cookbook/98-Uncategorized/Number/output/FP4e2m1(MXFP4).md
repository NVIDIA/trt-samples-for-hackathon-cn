# FP4e2m1(MXFP4) - not IEEE754

+ Sign:     $s$ ($p=1$ bit)
+ Exponent: $e$ ($q=2$ bits)
+ Mantissa: $m$ ($r=1$ bits)

+ Normal value ($01_2 \le e_2 \le 10_2$) (subscript 2 represents the base)

$$
\begin{equation}
\begin{aligned}
E &= e - \left( 2^{q-1} - 1 \right) = e - 1 \\
M &= m \cdot 2^{-r} = m \cdot 2^{-1} \\
value &= \left( -1 \right) ^ {s} 2 ^ {E} \left( 1 + M \right) = \left( -1 \right) ^ {s} 2 ^ {-2} 2 ^ {e} \left( m  + 2^{1} \right)
\end{aligned}
\end{equation}
$$

+ Subnormal value ($e_2 = 00_2$)

$$
\begin{equation}
\begin{aligned}
E &= 2 - 2^{q-1} = 0 \\
M &= m \cdot 2^{-r} = m \cdot 2^{-1} \\
value &= \left( -1 \right) ^ {s} 2 ^ {E} M = \left( -1 \right) ^ {s} 2^{0} m
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
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{0}$         | $+0$               | Positive Zero       |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{1}$         | $5.0000\times10^{-01}$ | Minimum Subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{1}$         | $5.0000\times10^{-01}$ | Maximum Subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{0}$         | $1.0000\times10^{+00}$ | Minimum Normal      |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{1}$     | $1 - 2^{-2}$       | Largest Number < 1  |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{0}$         | $1$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{1}$         | $1 + 2^{-1}$       | Smallest Number > 1 |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{0}$         | $2$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{1}$         | $3$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{11}\color{#0000FF}{1}$     | $6.0000\times10^{+00}$ | Maximum |
| $\color{#FF0000}{1}\color{#007F00}{11}\color{#0000FF}{1}$     | $-6.0000\times10^{+00}$ | Maximum negative    |
| $\color{#FF0000}{1}\color{#007F00}{00}\color{#0000FF}{0}$         | $-0$               | Negative Zero       |

+ All non-negative values

| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponen}\color{#0000FF}{Mantissa}$) | value |
| :-:        | :-:   |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{0}$     | 0.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{1}$     | 5.000e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{0}$     | 1.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{1}$     | 1.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{0}$     | 2.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{1}$     | 3.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{11}\color{#0000FF}{0}$     | 4.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{11}\color{#0000FF}{1}$     | 6.000e+00  |
