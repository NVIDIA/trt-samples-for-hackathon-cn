# S1E3M2(FP6e3m2) - not IEEE754

+ Sign:     $s$ ($p=1$ bit)
+ Exponent: $e$ ($q=3$ bits)
+ Mantissa: $m$ ($r=2$ bits)

+ Normal value ($001_2 \le e_2 \le 110_2$) (subscript 2 represents the base)

$$
\begin{equation}
\begin{aligned}
E &= e - \left( 2^{q-1} - 1 \right) = e - 3 \\
M &= m \cdot 2^{-r} = m \cdot 2^{-2} \\
value &= \left( -1 \right) ^ {s} 2 ^ {E} \left( 1 + M \right) = \left( -1 \right) ^ {s} 2 ^ {-5} 2 ^ {e} \left( m  + 2^{2} \right)
\end{aligned}
\end{equation}
$$

+ Subnormal value ($e_2 = 000_2$)

$$
\begin{equation}
\begin{aligned}
E &= 2 - 2^{q-1} = -2 \\
M &= m \cdot 2^{-r} = m \cdot 2^{-2} \\
value &= \left( -1 \right) ^ {s} 2 ^ {E} M = \left( -1 \right) ^ {s} 2^{-2} m
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

| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponen}\color{#0000FF}{Mantissa}$) | value          | comment          |
| :-:        | :-:            | :-:              |
| $\color{#FF0000}{0}\color{#007F00}{000}\color{#0000FF}{00}$         | $+0$               | Positive Zero       |
| $\color{#FF0000}{0}\color{#007F00}{000}\color{#0000FF}{01}$         | $6.2500\times10^{-02}$ | Minimum Subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{000}\color{#0000FF}{11}$         | $1.8750\times10^{-01}$ | Maximum Subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{001}\color{#0000FF}{00}$         | $2.5000\times10^{-01}$ | Minimum Normal      |
| $\color{#FF0000}{0}\color{#007F00}{010}\color{#0000FF}{11}$     | $1 - 2^{-3}$       | Largest Number < 1  |
| $\color{#FF0000}{0}\color{#007F00}{011}\color{#0000FF}{00}$         | $1$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{011}\color{#0000FF}{01}$         | $1 + 2^{-2}$       | Smallest Number > 1 |
| $\color{#FF0000}{0}\color{#007F00}{100}\color{#0000FF}{00}$         | $2$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{100}\color{#0000FF}{10}$         | $3$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{101}\color{#0000FF}{00}$     | $4$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{101}\color{#0000FF}{01}$     | $5$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{101}\color{#0000FF}{10}$     | $6$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{111}\color{#0000FF}{11}$     | $2.8000\times10^{+01}$ | Maximum |
| $\color{#FF0000}{1}\color{#007F00}{111}\color{#0000FF}{11}$     | $-2.8000\times10^{+01}$ | Maximum negative    |
| $\color{#FF0000}{1}\color{#007F00}{000}\color{#0000FF}{00}$         | $-0$               | Negative Zero       |
| $\color{#FF0000}{0}\color{#007F00}{001}\color{#0000FF}{01}$     | $3.1250\times10^{-01}$ | $\frac{1}{3}$      |

+ All non-negative values

| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponen}\color{#0000FF}{Mantissa}$) | value |
| :-:        | :-:   |
| $\color{#FF0000}{0}\color{#007F00}{000}\color{#0000FF}{00}$     | 0.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{000}\color{#0000FF}{01}$     | 6.250e-02  |
| $\color{#FF0000}{0}\color{#007F00}{000}\color{#0000FF}{10}$     | 1.250e-01  |
| $\color{#FF0000}{0}\color{#007F00}{000}\color{#0000FF}{11}$     | 1.875e-01  |
| $\color{#FF0000}{0}\color{#007F00}{001}\color{#0000FF}{00}$     | 2.500e-01  |
| $\color{#FF0000}{0}\color{#007F00}{001}\color{#0000FF}{01}$     | 3.125e-01  |
| $\color{#FF0000}{0}\color{#007F00}{001}\color{#0000FF}{10}$     | 3.750e-01  |
| $\color{#FF0000}{0}\color{#007F00}{001}\color{#0000FF}{11}$     | 4.375e-01  |
| $\color{#FF0000}{0}\color{#007F00}{010}\color{#0000FF}{00}$     | 5.000e-01  |
| $\color{#FF0000}{0}\color{#007F00}{010}\color{#0000FF}{01}$     | 6.250e-01  |
| $\color{#FF0000}{0}\color{#007F00}{010}\color{#0000FF}{10}$     | 7.500e-01  |
| $\color{#FF0000}{0}\color{#007F00}{010}\color{#0000FF}{11}$     | 8.750e-01  |
| $\color{#FF0000}{0}\color{#007F00}{011}\color{#0000FF}{00}$     | 1.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{011}\color{#0000FF}{01}$     | 1.250e+00  |
| $\color{#FF0000}{0}\color{#007F00}{011}\color{#0000FF}{10}$     | 1.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{011}\color{#0000FF}{11}$     | 1.750e+00  |
| $\color{#FF0000}{0}\color{#007F00}{100}\color{#0000FF}{00}$     | 2.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{100}\color{#0000FF}{01}$     | 2.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{100}\color{#0000FF}{10}$     | 3.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{100}\color{#0000FF}{11}$     | 3.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{101}\color{#0000FF}{00}$     | 4.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{101}\color{#0000FF}{01}$     | 5.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{101}\color{#0000FF}{10}$     | 6.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{101}\color{#0000FF}{11}$     | 7.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{110}\color{#0000FF}{00}$     | 8.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{110}\color{#0000FF}{01}$     | 1.000e+01  |
| $\color{#FF0000}{0}\color{#007F00}{110}\color{#0000FF}{10}$     | 1.200e+01  |
| $\color{#FF0000}{0}\color{#007F00}{110}\color{#0000FF}{11}$     | 1.400e+01  |
| $\color{#FF0000}{0}\color{#007F00}{111}\color{#0000FF}{00}$     | 1.600e+01  |
| $\color{#FF0000}{0}\color{#007F00}{111}\color{#0000FF}{01}$     | 2.000e+01  |
| $\color{#FF0000}{0}\color{#007F00}{111}\color{#0000FF}{10}$     | 2.400e+01  |
| $\color{#FF0000}{0}\color{#007F00}{111}\color{#0000FF}{11}$     | 2.800e+01  |
