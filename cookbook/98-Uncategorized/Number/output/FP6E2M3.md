# FP6E2M3 - not IEEE754

+ Sign:     $s$ ($p = 1$ bit)
+ Exponent: $e$ ($q = 2$ bit)
+ Mantissa: $m$ ($r = 3$ bit)

+ Normal value ($01_2 \le e_2 \le 10_2$) (2 in subscript represents the base)

$$
\begin{equation}
\begin{aligned}
E &= e - \left( 2 ^ {q-1} - 1 \right) = e - 1 \\
M &= m \cdot 2 ^ {-r} = m \cdot 2 ^ {-3} \\
\text{value} &= \left( -1 \right) ^ {s} 2 ^ {E} \left( 1 + M \right) = \left( -1 \right) ^ {s} 2 ^ {-4} 2 ^ {e} \left( m + 2 ^ {3} \right)
\end{aligned}
\end{equation}
$$

+ Subnormal value ($e_2 = 00_2$)

$$
\begin{equation}
\begin{aligned}
E &= 2 - 2 ^ {q-1} = -0 \\
M &= m \cdot 2 ^ {-r} = m \cdot 2 ^ {-3} \\
\text{value} &= \left( -1 \right) ^ {s} 2 ^ {E} M = \left( -1 \right) ^ {s} 2 ^ {-3} m
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
| $\color{#D62728}{0}\color{#2CA02C}{00}\color{#1F77B4}{000}$ | $+0$ | Positive Zero |
| $\color{#D62728}{0}\color{#2CA02C}{00}\color{#1F77B4}{001}$ | $1.250000\times10^{-01}$ | Minimum Subnormal |
| $\color{#D62728}{0}\color{#2CA02C}{00}\color{#1F77B4}{111}$ | $8.750000\times10^{-01}$ | Maximum Subnormal |
| $\color{#D62728}{0}\color{#2CA02C}{01}\color{#1F77B4}{000}$ | $1.000000\times10^{+00}$ | Minimum Normal |
| $\color{#D62728}{0}\color{#2CA02C}{00}\color{#1F77B4}{111}$     | $1 - 2^{-3}$       | Largest number < 1  |
| $\color{#D62728}{0}\color{#2CA02C}{01}\color{#1F77B4}{000}$ | $1$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{01}\color{#1F77B4}{001}$ | $1 + 2 ^ {-3}$ | Smallest number > 1 |
| $\color{#D62728}{0}\color{#2CA02C}{10}\color{#1F77B4}{000}$ | $2$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{10}\color{#1F77B4}{100}$ | $3$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{11}\color{#1F77B4}{000}$ | $4$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{11}\color{#1F77B4}{010}$ | $5$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{11}\color{#1F77B4}{100}$ | $6$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{11}\color{#1F77B4}{111}$ | $7.500000\times10^{+00}$ | Positive maximum |
| $\color{#D62728}{1}\color{#2CA02C}{11}\color{#1F77B4}{111}$ | $-7.500000\times10^{+00}$ | Negative maximum |
| $\color{#D62728}{1}\color{#2CA02C}{00}\color{#1F77B4}{000}$ | $-0$ | Negative Zero |

|$\color{#D62728}{S}$|$\color{#2CA02C}{E}$|$\color{#1F77B4}{M=000}$|$\color{#1F77B4}001$|$\color{#1F77B4}010$|$\color{#1F77B4}011$|$\color{#1F77B4}100$|$\color{#1F77B4}101$|$\color{#1F77B4}110$|$\color{#1F77B4}111$|
|:-:|:-:|-:|-:|-:|-:|-:|-:|-:|-:|
|$\color{#D62728}{0}$|$\color{#2CA02C}{00}$|0.000000e+00|1.250000e-01|2.500000e-01|3.750000e-01|5.000000e-01|6.250000e-01|7.500000e-01|8.750000e-01|
|$\color{#D62728}{0}$|$\color{#2CA02C}{01}$|1.000000e+00|1.125000e+00|1.250000e+00|1.375000e+00|1.500000e+00|1.625000e+00|1.750000e+00|1.875000e+00|
|$\color{#D62728}{0}$|$\color{#2CA02C}{10}$|2.000000e+00|2.250000e+00|2.500000e+00|2.750000e+00|3.000000e+00|3.250000e+00|3.500000e+00|3.750000e+00|
|$\color{#D62728}{0}$|$\color{#2CA02C}{11}$|4.000000e+00|4.500000e+00|5.000000e+00|5.500000e+00|6.000000e+00|6.500000e+00|7.000000e+00|7.500000e+00|
|$\color{#D62728}{1}$|$\color{#2CA02C}{00}$|-0.000000e+00|-1.250000e-01|-2.500000e-01|-3.750000e-01|-5.000000e-01|-6.250000e-01|-7.500000e-01|-8.750000e-01|
|$\color{#D62728}{1}$|$\color{#2CA02C}{01}$|-1.000000e+00|-1.125000e+00|-1.250000e+00|-1.375000e+00|-1.500000e+00|-1.625000e+00|-1.750000e+00|-1.875000e+00|
|$\color{#D62728}{1}$|$\color{#2CA02C}{10}$|-2.000000e+00|-2.250000e+00|-2.500000e+00|-2.750000e+00|-3.000000e+00|-3.250000e+00|-3.500000e+00|-3.750000e+00|
|$\color{#D62728}{1}$|$\color{#2CA02C}{11}$|-4.000000e+00|-4.500000e+00|-5.000000e+00|-5.500000e+00|-6.000000e+00|-6.500000e+00|-7.000000e+00|-7.500000e+00|
