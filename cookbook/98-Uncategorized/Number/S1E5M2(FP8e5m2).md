# S1E5M2(FP8e5m2) - IEEE754

+ Sign:     $s$ ($p=1$ bit)
+ Exponent: $e$ ($q=5$ bits)
+ Mantissa: $m$ ($r=2$ bits)

+ Normal value ($00001_2 \le e_2 \le 11110_2$) (subscript 2 represents the base)

$$
\begin{equation}
\begin{aligned}
E &= e - \left( 2^{q-1} - 1 \right) = e - 15 \\
M &= m \cdot 2^{-r} = m \cdot 2^{-2} \\
value &= \left( -1 \right) ^ {s} 2 ^ {E} \left( 1 + M \right) = \left( -1 \right) ^ {s} 2 ^ {-17} 2 ^ {e} \left( m  + 2^{2} \right)
\end{aligned}
\end{equation}
$$

+ Subnormal value ($e_2 = 00000_2$)

$$
\begin{equation}
\begin{aligned}
E &= 2 - 2^{q-1} = -14 \\
M &= m \cdot 2^{-r} = m \cdot 2^{-2} \\
value &= \left( -1 \right) ^ {s} 2 ^ {E} M = \left( -1 \right) ^ {s} 2^{-14} m
\end{aligned}
\end{equation}
$$

+ Special value

| Exponent  | Mantissa  | meaning         |
| :-:       | :-:       | :-:             |
| all 0     | all 0     | Signed Zero     |
| all 0     | not all 0 | Subnormal Value |
| all 1 | all 0     | Signed Infinity |
| all 1 | not all 0 | qNaN, sNaN      |

+ Examples

| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponen}\color{#0000FF}{Mantissa}$) | value          | comment          |
| :-:        | :-:            | :-:              |
| $\color{#FF0000}{0}\color{#007F00}{00000}\color{#0000FF}{00}$         | $+0$               | Positive Zero       |
| $\color{#FF0000}{0}\color{#007F00}{00000}\color{#0000FF}{01}$         | $1.5259\times10^{-05}$ | Minimum Subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{00000}\color{#0000FF}{11}$         | $4.5776\times10^{-05}$ | Maximum Subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{00001}\color{#0000FF}{00}$         | $6.1035\times10^{-05}$ | Minimum Normal      |
| $\color{#FF0000}{0}\color{#007F00}{01110}\color{#0000FF}{11}$     | $1 - 2^{-3}$       | Largest Number < 1  |
| $\color{#FF0000}{0}\color{#007F00}{01111}\color{#0000FF}{00}$         | $1$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{01111}\color{#0000FF}{01}$         | $1 + 2^{-2}$       | Smallest Number > 1 |
| $\color{#FF0000}{0}\color{#007F00}{10000}\color{#0000FF}{00}$         | $2$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{10000}\color{#0000FF}{10}$         | $3$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{10001}\color{#0000FF}{00}$     | $4$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{10001}\color{#0000FF}{01}$     | $5$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{10001}\color{#0000FF}{10}$     | $6$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{11110}\color{#0000FF}{11}$     | $5.7344\times10^{+04}$ | Maximum |
| $\color{#FF0000}{1}\color{#007F00}{11110}\color{#0000FF}{11}$     | $-5.7344\times10^{+04}$ | Maximum negative    |
| $\color{#FF0000}{1}\color{#007F00}{00000}\color{#0000FF}{00}$         | $-0$               | Negative Zero       |
| $\color{#FF0000}{0}\color{#007F00}{11111}\color{#0000FF}{00}$     | $+\infty$          | Positive Infinity   |
| $\color{#FF0000}{1}\color{#007F00}{11111}\color{#0000FF}{00}$     | $-\infty$          | Negative Infinity   |
| $\color{#FF0000}{0}\color{#007F00}{11111}\color{#0000FF}{01}$     | $NaN$              | Signalling NaN      |
| $\color{#FF0000}{0}\color{#007F00}{11111}\color{#0000FF}{11}$ | $NaN$              | Quiet NaN           |
| $\color{#FF0000}{0}\color{#007F00}{11111}\color{#0000FF}{11}$         | $NaN$              | NaN                 |
| $\color{#FF0000}{0}\color{#007F00}{01101}\color{#0000FF}{01}$     | $3.1250\times10^{-01}$ | $\frac{1}{3}$      |

+ All non-negative values

| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponen}\color{#0000FF}{Mantissa}$) | value |
| :-:        | :-:   |
| $\color{#FF0000}{0}\color{#007F00}{00000}\color{#0000FF}{00}$     | 0.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{00000}\color{#0000FF}{01}$     | 1.526e-05  |
| $\color{#FF0000}{0}\color{#007F00}{00000}\color{#0000FF}{10}$     | 3.052e-05  |
| $\color{#FF0000}{0}\color{#007F00}{00000}\color{#0000FF}{11}$     | 4.578e-05  |
| $\color{#FF0000}{0}\color{#007F00}{00001}\color{#0000FF}{00}$     | 6.104e-05  |
| $\color{#FF0000}{0}\color{#007F00}{00001}\color{#0000FF}{01}$     | 7.629e-05  |
| $\color{#FF0000}{0}\color{#007F00}{00001}\color{#0000FF}{10}$     | 9.155e-05  |
| $\color{#FF0000}{0}\color{#007F00}{00001}\color{#0000FF}{11}$     | 1.068e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00010}\color{#0000FF}{00}$     | 1.221e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00010}\color{#0000FF}{01}$     | 1.526e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00010}\color{#0000FF}{10}$     | 1.831e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00010}\color{#0000FF}{11}$     | 2.136e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00011}\color{#0000FF}{00}$     | 2.441e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00011}\color{#0000FF}{01}$     | 3.052e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00011}\color{#0000FF}{10}$     | 3.662e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00011}\color{#0000FF}{11}$     | 4.272e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00100}\color{#0000FF}{00}$     | 4.883e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00100}\color{#0000FF}{01}$     | 6.104e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00100}\color{#0000FF}{10}$     | 7.324e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00100}\color{#0000FF}{11}$     | 8.545e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00101}\color{#0000FF}{00}$     | 9.766e-04  |
| $\color{#FF0000}{0}\color{#007F00}{00101}\color{#0000FF}{01}$     | 1.221e-03  |
| $\color{#FF0000}{0}\color{#007F00}{00101}\color{#0000FF}{10}$     | 1.465e-03  |
| $\color{#FF0000}{0}\color{#007F00}{00101}\color{#0000FF}{11}$     | 1.709e-03  |
| $\color{#FF0000}{0}\color{#007F00}{00110}\color{#0000FF}{00}$     | 1.953e-03  |
| $\color{#FF0000}{0}\color{#007F00}{00110}\color{#0000FF}{01}$     | 2.441e-03  |
| $\color{#FF0000}{0}\color{#007F00}{00110}\color{#0000FF}{10}$     | 2.930e-03  |
| $\color{#FF0000}{0}\color{#007F00}{00110}\color{#0000FF}{11}$     | 3.418e-03  |
| $\color{#FF0000}{0}\color{#007F00}{00111}\color{#0000FF}{00}$     | 3.906e-03  |
| $\color{#FF0000}{0}\color{#007F00}{00111}\color{#0000FF}{01}$     | 4.883e-03  |
| $\color{#FF0000}{0}\color{#007F00}{00111}\color{#0000FF}{10}$     | 5.859e-03  |
| $\color{#FF0000}{0}\color{#007F00}{00111}\color{#0000FF}{11}$     | 6.836e-03  |
| $\color{#FF0000}{0}\color{#007F00}{01000}\color{#0000FF}{00}$     | 7.812e-03  |
| $\color{#FF0000}{0}\color{#007F00}{01000}\color{#0000FF}{01}$     | 9.766e-03  |
| $\color{#FF0000}{0}\color{#007F00}{01000}\color{#0000FF}{10}$     | 1.172e-02  |
| $\color{#FF0000}{0}\color{#007F00}{01000}\color{#0000FF}{11}$     | 1.367e-02  |
| $\color{#FF0000}{0}\color{#007F00}{01001}\color{#0000FF}{00}$     | 1.562e-02  |
| $\color{#FF0000}{0}\color{#007F00}{01001}\color{#0000FF}{01}$     | 1.953e-02  |
| $\color{#FF0000}{0}\color{#007F00}{01001}\color{#0000FF}{10}$     | 2.344e-02  |
| $\color{#FF0000}{0}\color{#007F00}{01001}\color{#0000FF}{11}$     | 2.734e-02  |
| $\color{#FF0000}{0}\color{#007F00}{01010}\color{#0000FF}{00}$     | 3.125e-02  |
| $\color{#FF0000}{0}\color{#007F00}{01010}\color{#0000FF}{01}$     | 3.906e-02  |
| $\color{#FF0000}{0}\color{#007F00}{01010}\color{#0000FF}{10}$     | 4.688e-02  |
| $\color{#FF0000}{0}\color{#007F00}{01010}\color{#0000FF}{11}$     | 5.469e-02  |
| $\color{#FF0000}{0}\color{#007F00}{01011}\color{#0000FF}{00}$     | 6.250e-02  |
| $\color{#FF0000}{0}\color{#007F00}{01011}\color{#0000FF}{01}$     | 7.812e-02  |
| $\color{#FF0000}{0}\color{#007F00}{01011}\color{#0000FF}{10}$     | 9.375e-02  |
| $\color{#FF0000}{0}\color{#007F00}{01011}\color{#0000FF}{11}$     | 1.094e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01100}\color{#0000FF}{00}$     | 1.250e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01100}\color{#0000FF}{01}$     | 1.562e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01100}\color{#0000FF}{10}$     | 1.875e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01100}\color{#0000FF}{11}$     | 2.188e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01101}\color{#0000FF}{00}$     | 2.500e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01101}\color{#0000FF}{01}$     | 3.125e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01101}\color{#0000FF}{10}$     | 3.750e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01101}\color{#0000FF}{11}$     | 4.375e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01110}\color{#0000FF}{00}$     | 5.000e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01110}\color{#0000FF}{01}$     | 6.250e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01110}\color{#0000FF}{10}$     | 7.500e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01110}\color{#0000FF}{11}$     | 8.750e-01  |
| $\color{#FF0000}{0}\color{#007F00}{01111}\color{#0000FF}{00}$     | 1.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{01111}\color{#0000FF}{01}$     | 1.250e+00  |
| $\color{#FF0000}{0}\color{#007F00}{01111}\color{#0000FF}{10}$     | 1.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{01111}\color{#0000FF}{11}$     | 1.750e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10000}\color{#0000FF}{00}$     | 2.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10000}\color{#0000FF}{01}$     | 2.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10000}\color{#0000FF}{10}$     | 3.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10000}\color{#0000FF}{11}$     | 3.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10001}\color{#0000FF}{00}$     | 4.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10001}\color{#0000FF}{01}$     | 5.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10001}\color{#0000FF}{10}$     | 6.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10001}\color{#0000FF}{11}$     | 7.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10010}\color{#0000FF}{00}$     | 8.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{10010}\color{#0000FF}{01}$     | 1.000e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10010}\color{#0000FF}{10}$     | 1.200e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10010}\color{#0000FF}{11}$     | 1.400e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10011}\color{#0000FF}{00}$     | 1.600e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10011}\color{#0000FF}{01}$     | 2.000e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10011}\color{#0000FF}{10}$     | 2.400e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10011}\color{#0000FF}{11}$     | 2.800e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10100}\color{#0000FF}{00}$     | 3.200e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10100}\color{#0000FF}{01}$     | 4.000e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10100}\color{#0000FF}{10}$     | 4.800e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10100}\color{#0000FF}{11}$     | 5.600e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10101}\color{#0000FF}{00}$     | 6.400e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10101}\color{#0000FF}{01}$     | 8.000e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10101}\color{#0000FF}{10}$     | 9.600e+01  |
| $\color{#FF0000}{0}\color{#007F00}{10101}\color{#0000FF}{11}$     | 1.120e+02  |
| $\color{#FF0000}{0}\color{#007F00}{10110}\color{#0000FF}{00}$     | 1.280e+02  |
| $\color{#FF0000}{0}\color{#007F00}{10110}\color{#0000FF}{01}$     | 1.600e+02  |
| $\color{#FF0000}{0}\color{#007F00}{10110}\color{#0000FF}{10}$     | 1.920e+02  |
| $\color{#FF0000}{0}\color{#007F00}{10110}\color{#0000FF}{11}$     | 2.240e+02  |
| $\color{#FF0000}{0}\color{#007F00}{10111}\color{#0000FF}{00}$     | 2.560e+02  |
| $\color{#FF0000}{0}\color{#007F00}{10111}\color{#0000FF}{01}$     | 3.200e+02  |
| $\color{#FF0000}{0}\color{#007F00}{10111}\color{#0000FF}{10}$     | 3.840e+02  |
| $\color{#FF0000}{0}\color{#007F00}{10111}\color{#0000FF}{11}$     | 4.480e+02  |
| $\color{#FF0000}{0}\color{#007F00}{11000}\color{#0000FF}{00}$     | 5.120e+02  |
| $\color{#FF0000}{0}\color{#007F00}{11000}\color{#0000FF}{01}$     | 6.400e+02  |
| $\color{#FF0000}{0}\color{#007F00}{11000}\color{#0000FF}{10}$     | 7.680e+02  |
| $\color{#FF0000}{0}\color{#007F00}{11000}\color{#0000FF}{11}$     | 8.960e+02  |
| $\color{#FF0000}{0}\color{#007F00}{11001}\color{#0000FF}{00}$     | 1.024e+03  |
| $\color{#FF0000}{0}\color{#007F00}{11001}\color{#0000FF}{01}$     | 1.280e+03  |
| $\color{#FF0000}{0}\color{#007F00}{11001}\color{#0000FF}{10}$     | 1.536e+03  |
| $\color{#FF0000}{0}\color{#007F00}{11001}\color{#0000FF}{11}$     | 1.792e+03  |
| $\color{#FF0000}{0}\color{#007F00}{11010}\color{#0000FF}{00}$     | 2.048e+03  |
| $\color{#FF0000}{0}\color{#007F00}{11010}\color{#0000FF}{01}$     | 2.560e+03  |
| $\color{#FF0000}{0}\color{#007F00}{11010}\color{#0000FF}{10}$     | 3.072e+03  |
| $\color{#FF0000}{0}\color{#007F00}{11010}\color{#0000FF}{11}$     | 3.584e+03  |
| $\color{#FF0000}{0}\color{#007F00}{11011}\color{#0000FF}{00}$     | 4.096e+03  |
| $\color{#FF0000}{0}\color{#007F00}{11011}\color{#0000FF}{01}$     | 5.120e+03  |
| $\color{#FF0000}{0}\color{#007F00}{11011}\color{#0000FF}{10}$     | 6.144e+03  |
| $\color{#FF0000}{0}\color{#007F00}{11011}\color{#0000FF}{11}$     | 7.168e+03  |
| $\color{#FF0000}{0}\color{#007F00}{11100}\color{#0000FF}{00}$     | 8.192e+03  |
| $\color{#FF0000}{0}\color{#007F00}{11100}\color{#0000FF}{01}$     | 1.024e+04  |
| $\color{#FF0000}{0}\color{#007F00}{11100}\color{#0000FF}{10}$     | 1.229e+04  |
| $\color{#FF0000}{0}\color{#007F00}{11100}\color{#0000FF}{11}$     | 1.434e+04  |
| $\color{#FF0000}{0}\color{#007F00}{11101}\color{#0000FF}{00}$     | 1.638e+04  |
| $\color{#FF0000}{0}\color{#007F00}{11101}\color{#0000FF}{01}$     | 2.048e+04  |
| $\color{#FF0000}{0}\color{#007F00}{11101}\color{#0000FF}{10}$     | 2.458e+04  |
| $\color{#FF0000}{0}\color{#007F00}{11101}\color{#0000FF}{11}$     | 2.867e+04  |
| $\color{#FF0000}{0}\color{#007F00}{11110}\color{#0000FF}{00}$     | 3.277e+04  |
| $\color{#FF0000}{0}\color{#007F00}{11110}\color{#0000FF}{01}$     | 4.096e+04  |
| $\color{#FF0000}{0}\color{#007F00}{11110}\color{#0000FF}{10}$     | 4.915e+04  |
| $\color{#FF0000}{0}\color{#007F00}{11110}\color{#0000FF}{11}$     | 5.734e+04  |
| $\color{#FF0000}{0}\color{#007F00}{11111}\color{#0000FF}{00}$     | inf  |
| $\color{#FF0000}{0}\color{#007F00}{11111}\color{#0000FF}{01}$     | nan  |
| $\color{#FF0000}{0}\color{#007F00}{11111}\color{#0000FF}{10}$     | nan  |
| $\color{#FF0000}{0}\color{#007F00}{11111}\color{#0000FF}{11}$     | nan  |
