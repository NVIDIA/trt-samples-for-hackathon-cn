# FP6e2m3 - not IEEE754

+ Sign:     $s$ ($p=1$ bit)
+ Exponent: $e$ ($q=2$ bits)
+ Mantissa: $m$ ($r=3$ bits)

+ Normal value ($01_2 \le e_2 \le 10_2$) (subscript 2 represents the base)

$$
\begin{equation}
\begin{aligned}
E &= e - \left( 2^{q-1} - 1 \right) = e - 1 \\
M &= m \cdot 2^{-r} = m \cdot 2^{-3} \\
value &= \left( -1 \right) ^ {s} 2 ^ {E} \left( 1 + M \right) = \left( -1 \right) ^ {s} 2 ^ {-4} 2 ^ {e} \left( m  + 2^{3} \right)
\end{aligned}
\end{equation}
$$

+ Subnormal value ($e_2 = 00_2$)

$$
\begin{equation}
\begin{aligned}
E &= 2 - 2^{q-1} = 0 \\
M &= m \cdot 2^{-r} = m \cdot 2^{-3} \\
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
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{000}$         | $+0$               | Positive Zero       |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{001}$         | $1.2500\times10^{-01}$ | Minimum Subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{111}$         | $8.7500\times10^{-01}$ | Maximum Subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{000}$         | $1.0000\times10^{+00}$ | Minimum Normal      |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{111}$     | $1 - 2^{-3}$       | Largest Number < 1  |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{000}$         | $1$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{001}$         | $1 + 2^{-3}$       | Smallest Number > 1 |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{000}$         | $2$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{100}$         | $3$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{11}\color{#0000FF}{111}$     | $7.5000\times10^{+00}$ | Maximum |
| $\color{#FF0000}{1}\color{#007F00}{11}\color{#0000FF}{111}$     | $-7.5000\times10^{+00}$ | Maximum negative    |
| $\color{#FF0000}{1}\color{#007F00}{00}\color{#0000FF}{000}$         | $-0$               | Negative Zero       |

+ All non-negative values

| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponen}\color{#0000FF}{Mantissa}$) | value |
| :-:        | :-:   |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{000}$     | 0.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{001}$     | 1.250e-01  |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{010}$     | 2.500e-01  |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{011}$     | 3.750e-01  |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{100}$     | 5.000e-01  |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{101}$     | 6.250e-01  |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{110}$     | 7.500e-01  |
| $\color{#FF0000}{0}\color{#007F00}{00}\color{#0000FF}{111}$     | 8.750e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{000}$     | 1.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{001}$     | 1.125e+00  |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{010}$     | 1.250e+00  |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{011}$     | 1.375e+00  |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{100}$     | 1.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{101}$     | 1.625e+00  |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{110}$     | 1.750e+00  |
| $\color{#FF0000}{0}\color{#007F00}{01}\color{#0000FF}{111}$     | 1.875e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{000}$     | 2.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{001}$     | 2.250e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{010}$     | 2.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{011}$     | 2.750e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{100}$     | 3.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{101}$     | 3.250e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{110}$     | 3.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10}\color{#0000FF}{111}$     | 3.750e+00  |
| $\color{#FF0000}{0}\color{#007F00}{11}\color{#0000FF}{000}$     | 4.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{11}\color{#0000FF}{001}$     | 4.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{11}\color{#0000FF}{010}$     | 5.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{11}\color{#0000FF}{011}$     | 5.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{11}\color{#0000FF}{100}$     | 6.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{11}\color{#0000FF}{101}$     | 6.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{11}\color{#0000FF}{110}$     | 7.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{11}\color{#0000FF}{111}$     | 7.500e+00  |
