# FP4E0M3 - not IEEE754

+ Sign:     $s$ ($p = 1$ bit)
+ Exponent: $e$ ($q = 0$ bit)
+ Mantissa: $m$ ($r = 3$ bit)

+ Value

$$
\begin{equation}
\begin{aligned}
E &= 0 \\
M &= m \cdot 2 ^ {-r} = m \cdot 2 ^ {-3} \\
\text{value} &= \left( -1 \right) ^ {s} 2 ^ {E} M = \left( -1 \right) ^ {s} m \cdot 2 ^ {-3}
\end{aligned}
\end{equation}
$$


|$\color{#D62728}{S}$|$\color{#2CA02C}{E}$|$\color{#1F77B4}{M=000}$|$\color{#1F77B4}001$|$\color{#1F77B4}010$|$\color{#1F77B4}011$|$\color{#1F77B4}100$|$\color{#1F77B4}101$|$\color{#1F77B4}110$|$\color{#1F77B4}111$|
|:-:|:-:|-:|-:|-:|-:|-:|-:|-:|-:|
|$\color{#D62728}{0}$|$\color{#2CA02C}{0}$|0.000000e+00|1.250000e-01|2.500000e-01|3.750000e-01|5.000000e-01|6.250000e-01|7.500000e-01|8.750000e-01|
|$\color{#D62728}{1}$|$\color{#2CA02C}{0}$|-0.000000e+00|-1.250000e-01|-2.500000e-01|-3.750000e-01|-5.000000e-01|-6.250000e-01|-7.500000e-01|-8.750000e-01|
