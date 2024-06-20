# S1E4M3(FP8e4m3) - not IEEE754

+ Sign:     $s$ ($p=1$ bit)
+ Exponent: $e$ ($q=4$ bits)
+ Mantissa: $m$ ($r=3$ bits)

+ Normal value ($0001_2 \le e_2 \le 1110_2$) (subscript 2 represents the base)

$$
\begin{equation}
\begin{aligned}
E &= e - \left( 2^{q-1} - 1 \right) = e - 7 \\
M &= m \cdot 2^{-r} = m \cdot 2^{-3} \\
value &= \left( -1 \right) ^ {s} 2 ^ {E} \left( 1 + M \right) = \left( -1 \right) ^ {s} 2 ^ {-10} 2 ^ {e} \left( m  + 2^{3} \right)
\end{aligned}
\end{equation}
$$

+ Subnormal value ($e_2 = 0000_2$)

$$
\begin{equation}
\begin{aligned}
E &= 2 - 2^{q-1} = -6 \\
M &= m \cdot 2^{-r} = m \cdot 2^{-3} \\
value &= \left( -1 \right) ^ {s} 2 ^ {E} M = \left( -1 \right) ^ {s} 2^{-6} m
\end{aligned}
\end{equation}
$$

+ Special value

| Exponent  | Mantissa  | meaning         |
| :-:       | :-:       | :-:             |
| all 0     | all 0     | Signed Zero     |
| all 0     | not all 0 | Subnormal Value |
| all 1 | all 0     | Normal value    |
| all 1 | not all 0 | qNaN, sNaN      |

+ Examples

| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponen}\color{#0000FF}{Mantissa}$) | value          | comment          |
| :-:        | :-:            | :-:              |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{000}$         | $+0$               | Positive Zero       |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{001}$         | $1.9531\times10^{-03}$ | Minimum Subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{111}$         | $1.3672\times10^{-02}$ | Maximum Subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{0001}\color{#0000FF}{000}$         | $1.5625\times10^{-02}$ | Minimum Normal      |
| $\color{#FF0000}{0}\color{#007F00}{0110}\color{#0000FF}{111}$     | $1 - 2^{-4}$       | Largest Number < 1  |
| $\color{#FF0000}{0}\color{#007F00}{0111}\color{#0000FF}{000}$         | $1$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{0111}\color{#0000FF}{001}$         | $1 + 2^{-3}$       | Smallest Number > 1 |
| $\color{#FF0000}{0}\color{#007F00}{1000}\color{#0000FF}{000}$         | $2$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{1000}\color{#0000FF}{100}$         | $3$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{000}$     | $4$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{010}$     | $5$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{100}$     | $6$                |                     |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{110}$     | $4.4800\times10^{+02}$ | Maximum |
| $\color{#FF0000}{1}\color{#007F00}{1111}\color{#0000FF}{110}$     | $-4.4800\times10^{+02}$ | Maximum negative    |
| $\color{#FF0000}{1}\color{#007F00}{0000}\color{#0000FF}{000}$         | $-0$               | Negative Zero       |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{111}$         | $NaN$              | NaN                 |
| $\color{#FF0000}{0}\color{#007F00}{0101}\color{#0000FF}{011}$     | $3.4375\times10^{-01}$ | $\frac{1}{3}$      |

+ All non-negative values

| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponen}\color{#0000FF}{Mantissa}$) | value |
| :-:        | :-:   |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{000}$     | 0.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{001}$     | 1.953e-03  |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{010}$     | 3.906e-03  |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{011}$     | 5.859e-03  |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{100}$     | 7.812e-03  |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{101}$     | 9.766e-03  |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{110}$     | 1.172e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{111}$     | 1.367e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0001}\color{#0000FF}{000}$     | 1.562e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0001}\color{#0000FF}{001}$     | 1.758e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0001}\color{#0000FF}{010}$     | 1.953e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0001}\color{#0000FF}{011}$     | 2.148e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0001}\color{#0000FF}{100}$     | 2.344e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0001}\color{#0000FF}{101}$     | 2.539e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0001}\color{#0000FF}{110}$     | 2.734e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0001}\color{#0000FF}{111}$     | 2.930e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0010}\color{#0000FF}{000}$     | 3.125e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0010}\color{#0000FF}{001}$     | 3.516e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0010}\color{#0000FF}{010}$     | 3.906e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0010}\color{#0000FF}{011}$     | 4.297e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0010}\color{#0000FF}{100}$     | 4.688e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0010}\color{#0000FF}{101}$     | 5.078e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0010}\color{#0000FF}{110}$     | 5.469e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0010}\color{#0000FF}{111}$     | 5.859e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0011}\color{#0000FF}{000}$     | 6.250e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0011}\color{#0000FF}{001}$     | 7.031e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0011}\color{#0000FF}{010}$     | 7.812e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0011}\color{#0000FF}{011}$     | 8.594e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0011}\color{#0000FF}{100}$     | 9.375e-02  |
| $\color{#FF0000}{0}\color{#007F00}{0011}\color{#0000FF}{101}$     | 1.016e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0011}\color{#0000FF}{110}$     | 1.094e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0011}\color{#0000FF}{111}$     | 1.172e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0100}\color{#0000FF}{000}$     | 1.250e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0100}\color{#0000FF}{001}$     | 1.406e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0100}\color{#0000FF}{010}$     | 1.562e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0100}\color{#0000FF}{011}$     | 1.719e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0100}\color{#0000FF}{100}$     | 1.875e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0100}\color{#0000FF}{101}$     | 2.031e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0100}\color{#0000FF}{110}$     | 2.188e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0100}\color{#0000FF}{111}$     | 2.344e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0101}\color{#0000FF}{000}$     | 2.500e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0101}\color{#0000FF}{001}$     | 2.812e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0101}\color{#0000FF}{010}$     | 3.125e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0101}\color{#0000FF}{011}$     | 3.438e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0101}\color{#0000FF}{100}$     | 3.750e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0101}\color{#0000FF}{101}$     | 4.062e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0101}\color{#0000FF}{110}$     | 4.375e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0101}\color{#0000FF}{111}$     | 4.688e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0110}\color{#0000FF}{000}$     | 5.000e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0110}\color{#0000FF}{001}$     | 5.625e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0110}\color{#0000FF}{010}$     | 6.250e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0110}\color{#0000FF}{011}$     | 6.875e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0110}\color{#0000FF}{100}$     | 7.500e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0110}\color{#0000FF}{101}$     | 8.125e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0110}\color{#0000FF}{110}$     | 8.750e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0110}\color{#0000FF}{111}$     | 9.375e-01  |
| $\color{#FF0000}{0}\color{#007F00}{0111}\color{#0000FF}{000}$     | 1.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{0111}\color{#0000FF}{001}$     | 1.125e+00  |
| $\color{#FF0000}{0}\color{#007F00}{0111}\color{#0000FF}{010}$     | 1.250e+00  |
| $\color{#FF0000}{0}\color{#007F00}{0111}\color{#0000FF}{011}$     | 1.375e+00  |
| $\color{#FF0000}{0}\color{#007F00}{0111}\color{#0000FF}{100}$     | 1.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{0111}\color{#0000FF}{101}$     | 1.625e+00  |
| $\color{#FF0000}{0}\color{#007F00}{0111}\color{#0000FF}{110}$     | 1.750e+00  |
| $\color{#FF0000}{0}\color{#007F00}{0111}\color{#0000FF}{111}$     | 1.875e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1000}\color{#0000FF}{000}$     | 2.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1000}\color{#0000FF}{001}$     | 2.250e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1000}\color{#0000FF}{010}$     | 2.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1000}\color{#0000FF}{011}$     | 2.750e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1000}\color{#0000FF}{100}$     | 3.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1000}\color{#0000FF}{101}$     | 3.250e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1000}\color{#0000FF}{110}$     | 3.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1000}\color{#0000FF}{111}$     | 3.750e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{000}$     | 4.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{001}$     | 4.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{010}$     | 5.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{011}$     | 5.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{100}$     | 6.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{101}$     | 6.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{110}$     | 7.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{111}$     | 7.500e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1010}\color{#0000FF}{000}$     | 8.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1010}\color{#0000FF}{001}$     | 9.000e+00  |
| $\color{#FF0000}{0}\color{#007F00}{1010}\color{#0000FF}{010}$     | 1.000e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1010}\color{#0000FF}{011}$     | 1.100e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1010}\color{#0000FF}{100}$     | 1.200e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1010}\color{#0000FF}{101}$     | 1.300e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1010}\color{#0000FF}{110}$     | 1.400e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1010}\color{#0000FF}{111}$     | 1.500e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1011}\color{#0000FF}{000}$     | 1.600e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1011}\color{#0000FF}{001}$     | 1.800e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1011}\color{#0000FF}{010}$     | 2.000e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1011}\color{#0000FF}{011}$     | 2.200e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1011}\color{#0000FF}{100}$     | 2.400e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1011}\color{#0000FF}{101}$     | 2.600e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1011}\color{#0000FF}{110}$     | 2.800e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1011}\color{#0000FF}{111}$     | 3.000e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1100}\color{#0000FF}{000}$     | 3.200e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1100}\color{#0000FF}{001}$     | 3.600e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1100}\color{#0000FF}{010}$     | 4.000e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1100}\color{#0000FF}{011}$     | 4.400e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1100}\color{#0000FF}{100}$     | 4.800e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1100}\color{#0000FF}{101}$     | 5.200e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1100}\color{#0000FF}{110}$     | 5.600e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1100}\color{#0000FF}{111}$     | 6.000e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1101}\color{#0000FF}{000}$     | 6.400e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1101}\color{#0000FF}{001}$     | 7.200e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1101}\color{#0000FF}{010}$     | 8.000e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1101}\color{#0000FF}{011}$     | 8.800e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1101}\color{#0000FF}{100}$     | 9.600e+01  |
| $\color{#FF0000}{0}\color{#007F00}{1101}\color{#0000FF}{101}$     | 1.040e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1101}\color{#0000FF}{110}$     | 1.120e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1101}\color{#0000FF}{111}$     | 1.200e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1110}\color{#0000FF}{000}$     | 1.280e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1110}\color{#0000FF}{001}$     | 1.440e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1110}\color{#0000FF}{010}$     | 1.600e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1110}\color{#0000FF}{011}$     | 1.760e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1110}\color{#0000FF}{100}$     | 1.920e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1110}\color{#0000FF}{101}$     | 2.080e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1110}\color{#0000FF}{110}$     | 2.240e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1110}\color{#0000FF}{111}$     | 2.400e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{000}$     | 2.560e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{001}$     | 2.880e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{010}$     | 3.200e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{011}$     | 3.520e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{100}$     | 3.840e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{101}$     | 4.160e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{110}$     | 4.480e+02  |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{111}$     | nan  |
