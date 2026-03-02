# FP4E2M1 (MXFP4 or NVFP4) - not IEEE754

+ Sign:     $s$ ($p = 1$ bit)
+ Exponent: $e$ ($q = 2$ bit)
+ Mantissa: $m$ ($r = 1$ bit)

+ Normal value ($01_2 \le e_2 \le 10_2$) (2 in subscript represents the base)

$$
\begin{equation}
\begin{aligned}
E &= e - \left( 2 ^ {q-1} - 1 \right) = e - 1 \\
M &= m \cdot 2 ^ {-r} = m \cdot 2 ^ {-1} \\
\text{value} &= \left( -1 \right) ^ {s} 2 ^ {E} \left( 1 + M \right) = \left( -1 \right) ^ {s} 2 ^ {-2} 2 ^ {e} \left( m + 2 ^ {1} \right)
\end{aligned}
\end{equation}
$$

+ Subnormal value ($e_2 = 00_2$)

$$
\begin{equation}
\begin{aligned}
E &= 2 - 2 ^ {q-1} = -0 \\
M &= m \cdot 2 ^ {-r} = m \cdot 2 ^ {-1} \\
\text{value} &= \left( -1 \right) ^ {s} 2 ^ {E} M = \left( -1 \right) ^ {s} 2 ^ {-1} m
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
| $\color{#D62728}{0}\color{#2CA02C}{00}\color{#1F77B4}{0}$ | $+0$ | Positive Zero |
| $\color{#D62728}{0}\color{#2CA02C}{00}\color{#1F77B4}{1}$ | $5.000000\times10^{-01}$ | Minimum Subnormal |
| $\color{#D62728}{0}\color{#2CA02C}{00}\color{#1F77B4}{1}$ | $5.000000\times10^{-01}$ | Maximum Subnormal |
| $\color{#D62728}{0}\color{#2CA02C}{01}\color{#1F77B4}{0}$ | $1.000000\times10^{+00}$ | Minimum Normal |
| $\color{#D62728}{0}\color{#2CA02C}{00}\color{#1F77B4}{1}$ | $1 - 2 ^ {-2}$ | Largest number < 1 |
| $\color{#D62728}{0}\color{#2CA02C}{01}\color{#1F77B4}{0}$ | $1$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{01}\color{#1F77B4}{1}$ | $1 + 2 ^ {-1}$ | Smallest number > 1 |
| $\color{#D62728}{0}\color{#2CA02C}{10}\color{#1F77B4}{0}$ | $2$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{10}\color{#1F77B4}{1}$ | $3$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{11}\color{#1F77B4}{0}$ | $4$ |  |
| $\color{#D62728}{0}\color{#2CA02C}{11}\color{#1F77B4}{1}$ | $6.000000\times10^{+00}$ | Positive maximum |
| $\color{#D62728}{1}\color{#2CA02C}{11}\color{#1F77B4}{1}$ | $-6.000000\times10^{+00}$ | Negative maximum |
| $\color{#D62728}{1}\color{#2CA02C}{00}\color{#1F77B4}{0}$ | $-0$ | Negative Zero |

|$\color{#D62728}{S}$|$\color{#2CA02C}{E}$|$\color{#1F77B4}{M=0}$|$\color{#1F77B4}1$|
|:-:|:-:|-:|-:|
|$\color{#D62728}{0}$|$\color{#2CA02C}{00}$|0.000000e+00|5.000000e-01|
|$\color{#D62728}{0}$|$\color{#2CA02C}{01}$|1.000000e+00|1.500000e+00|
|$\color{#D62728}{0}$|$\color{#2CA02C}{10}$|2.000000e+00|3.000000e+00|
|$\color{#D62728}{0}$|$\color{#2CA02C}{11}$|4.000000e+00|6.000000e+00|
|$\color{#D62728}{1}$|$\color{#2CA02C}{00}$|-0.000000e+00|-5.000000e-01|
|$\color{#D62728}{1}$|$\color{#2CA02C}{01}$|-1.000000e+00|-1.500000e+00|
|$\color{#D62728}{1}$|$\color{#2CA02C}{10}$|-2.000000e+00|-3.000000e+00|
|$\color{#D62728}{1}$|$\color{#2CA02C}{11}$|-4.000000e+00|-6.000000e+00|
