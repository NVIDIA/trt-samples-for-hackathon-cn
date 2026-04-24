# FP6E3M2 - not IEEE754

+ Sign:     $s$ ($p = 1$ bit)
+ Exponent: $e$ ($q = 3$ bit)
+ Mantissa: $m$ ($r = 2$ bit)

+ Normal value ($001_2 \le e_2 \le 110_2$) (2 in subscript represents the base)

$$
\begin{equation}
\begin{aligned}
E &= e - \left( 2 ^ {q-1} - 1 \right) = e - 3 \\
M &= m \cdot 2 ^ {-r} = m \cdot 2 ^ {-2} \\
\text{value} &= \left( -1 \right) ^ {s} 2 ^ {E} \left( 1 + M \right) = \left( -1 \right) ^ {s} 2 ^ {-5} 2 ^ {e} \left( m + 2 ^ {2} \right)
\end{aligned}
\end{equation}
$$

+ Subnormal value ($e_2 = 000_2$)

$$
\begin{equation}
\begin{aligned}
E &= 2 - 2 ^ {q-1} = -2 \\
M &= m \cdot 2 ^ {-r} = m \cdot 2 ^ {-2} \\
\text{value} &= \left( -1 \right) ^ {s} 2 ^ {E} M = \left( -1 \right) ^ {s} 2 ^ {-4} m
\end{aligned}
\end{equation}
$$

+ Special value
  - **No bits pattern for infinity.**
  - **No bits pattern for NaN.**

| Exponent | Mantissa | meaning |
|:-:|:-:|:-:|
| all 0 | all 0 | Signed Zero |
| all 0 | not all 0 | Subnormal Value |
| all 1 | all 0 | Normal value |
| all 1 | not all 0 | Normal value |

+ Examples

| Number($\color{#D62728}{Sign}\color{#2CA02C}{Exponent}\color{#1F77B4}{Mantissa}$) | value | comment |
|:-:|:-:|:-:|
| $\color{#D62728}{0}\color{#2CA02C}{000}\color{#1F77B4}{00}$ | $+0$ | Positive Zero |
| $\color{#D62728}{0}\color{#2CA02C}{000}\color{#1F77B4}{01}$ | $6.250000\times10^{-02}$ | Minimum Subnormal |
| $\color{#D62728}{0}\color{#2CA02C}{000}\color{#1F77B4}{11}$ | $1.875000\times10^{-01}$ | Maximum Subnormal |
| $\color{#D62728}{0}\color{#2CA02C}{001}\color{#1F77B4}{00}$ | $2.500000\times10^{-01}$ | Minimum Normal |
| $\color{#D62728}{0}\color{#2CA02C}{010}\color{#1F77B4}{11}$ | $1 - 2 ^ {-3}$ | Largest number < 1 |
| $\color{#D62728}{0}\color{#2CA02C}{011}\color{#1F77B4}{00}$ | $1$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{011}\color{#1F77B4}{01}$ | $1 + 2 ^ {-2}$ | Smallest number > 1 |
| $\color{#D62728}{0}\color{#2CA02C}{100}\color{#1F77B4}{00}$ | $2$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{100}\color{#1F77B4}{10}$ | $3$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{101}\color{#1F77B4}{00}$ | $4$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{101}\color{#1F77B4}{01}$ | $5$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{101}\color{#1F77B4}{10}$ | $6$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{111}\color{#1F77B4}{11}$ | $2.800000\times10^{+01}$ | Positive maximum |
| $\color{#D62728}{1}\color{#2CA02C}{111}\color{#1F77B4}{11}$ | $-2.800000\times10^{+01}$ | Negative maximum |
| $\color{#D62728}{1}\color{#2CA02C}{000}\color{#1F77B4}{00}$ | $-0$ | Negative Zero |
| $\color{#D62728}{0}\color{#2CA02C}{001}\color{#1F77B4}{01}$ | $3.125000\times10^{-01}$ | $\frac{1}{3}$ |

|$\color{#D62728}{S}$|$\color{#2CA02C}{E}$|$\color{#1F77B4}{M=00}$|$\color{#1F77B4}01$|$\color{#1F77B4}10$|$\color{#1F77B4}11$|
|:-:|:-:|-:|-:|-:|-:|
|$\color{#D62728}{0}$|$\color{#2CA02C}{000}$|0.000000e+00|6.250000e-02|1.250000e-01|1.875000e-01|
|$\color{#D62728}{0}$|$\color{#2CA02C}{001}$|2.500000e-01|3.125000e-01|3.750000e-01|4.375000e-01|
|$\color{#D62728}{0}$|$\color{#2CA02C}{010}$|5.000000e-01|6.250000e-01|7.500000e-01|8.750000e-01|
|$\color{#D62728}{0}$|$\color{#2CA02C}{011}$|1.000000e+00|1.250000e+00|1.500000e+00|1.750000e+00|
|$\color{#D62728}{0}$|$\color{#2CA02C}{100}$|2.000000e+00|2.500000e+00|3.000000e+00|3.500000e+00|
|$\color{#D62728}{0}$|$\color{#2CA02C}{101}$|4.000000e+00|5.000000e+00|6.000000e+00|7.000000e+00|
|$\color{#D62728}{0}$|$\color{#2CA02C}{110}$|8.000000e+00|1.000000e+01|1.200000e+01|1.400000e+01|
|$\color{#D62728}{0}$|$\color{#2CA02C}{111}$|1.600000e+01|2.000000e+01|2.400000e+01|2.800000e+01|
|$\color{#D62728}{1}$|$\color{#2CA02C}{000}$|-0.000000e+00|-6.250000e-02|-1.250000e-01|-1.875000e-01|
|$\color{#D62728}{1}$|$\color{#2CA02C}{001}$|-2.500000e-01|-3.125000e-01|-3.750000e-01|-4.375000e-01|
|$\color{#D62728}{1}$|$\color{#2CA02C}{010}$|-5.000000e-01|-6.250000e-01|-7.500000e-01|-8.750000e-01|
|$\color{#D62728}{1}$|$\color{#2CA02C}{011}$|-1.000000e+00|-1.250000e+00|-1.500000e+00|-1.750000e+00|
|$\color{#D62728}{1}$|$\color{#2CA02C}{100}$|-2.000000e+00|-2.500000e+00|-3.000000e+00|-3.500000e+00|
|$\color{#D62728}{1}$|$\color{#2CA02C}{101}$|-4.000000e+00|-5.000000e+00|-6.000000e+00|-7.000000e+00|
|$\color{#D62728}{1}$|$\color{#2CA02C}{110}$|-8.000000e+00|-1.000000e+01|-1.200000e+01|-1.400000e+01|
|$\color{#D62728}{1}$|$\color{#2CA02C}{111}$|-1.600000e+01|-2.000000e+01|-2.400000e+01|-2.800000e+01|
