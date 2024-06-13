# S1E5M2B0(FP8e5m2) - IEEE754

+ SignBit ($s$): 1
+ Exponent ($k$): 5
+ Mantissa ($n$): 2
+ Bias ($b$): 0

+ Special value

| Exponent   | all 0             | not all 0         |
| :-:        | :-:               | :-:               |
| $e = 00000_2$ | Signed Zero       | Subnormal Value   |
| $e = 11111_2$ | Signed Infinity   |       NaN         |

+ Normal value ($00001_2 \le e_2 \le 11110_2$)

$$
\begin{equation}
\begin{aligned}
E &= e_{10} - \left( 2^{k-1} - 1 \right) \\
M &= f_{10} \cdot 2^{-n} \\
value &= \left(-1\right)^{s}2^{E}\left(1+M\right)
\end{aligned}
\end{equation}
$$

+ Subnormal value ($e_2 = 00000_2$)

$$
\begin{equation}
\begin{aligned}
E &= 2 - 2^{k-1} = -14 \\
M &= f_{10} \cdot 2^{-n} \\
value &= \left(-1\right)^{s}2^{E}M
\end{aligned}
\end{equation}
$$

+ Examples

| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponent}\color{#0000FF}{Mantissa}$)  | value                 |        comment        |
| :-:                                                                                | :-:                   | :-:                   |
| $\color{#FF0000}{0}\color{#007F00}{00000}\color{#0000FF}{00}$                         | $+0$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{00000}\color{#0000FF}{01}$                        | $1.5259\times10^{-05}$    |   Minimum subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{00000}\color{#0000FF}{11}$                         | $4.5776\times10^{-05}$    |   Maximum subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{00001}\color{#0000FF}{00}$                        | $6.1035\times10^{-05}$    |    Minimum normal     |
| $\color{#FF0000}{0}\color{#007F00}{01110}\color{#0000FF}{11}$                   |  $1 - 2^{-3}$        |  largest number < 1   |
| $\color{#FF0000}{0}\color{#007F00}{01111}\color{#0000FF}{00}$                        |  $1$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{01111}\color{#0000FF}{01}$                       |  $1 + 2^{-2}$        |  smallest number > 1  |
| $\color{#FF0000}{0}\color{#007F00}{10000}\color{#0000FF}{00}$                        |  $2$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{10000}\color{#0000FF}{10}$                       |  $3$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{10001}\color{#0000FF}{00}$                   |  $4$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{10001}\color{#0000FF}{01}$             |  $5$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{10001}\color{#0000FF}{10}$             |  $6$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{11110}\color{#0000FF}{11}$                    | $5.7344\times10^{+04}$    |        Maximum        |
| $\color{#FF0000}{1}\color{#007F00}{11110}\color{#0000FF}{11}$                    | $-5.7344\times10^{+04}$    |     Maximum negative  |
| $\color{#FF0000}{1}\color{#007F00}{00000}\color{#0000FF}{00}$                         | $-0$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{11111}\color{#0000FF}{00}$                         | $+\infty$             |   positive infinity   |
| $\color{#FF0000}{1}\color{#007F00}{11111}\color{#0000FF}{00}$                         | $-\infty$             |   negative infinity   |
| $\color{#FF0000}{0}\color{#007F00}{11111}\color{#0000FF}{01}$                        | $NaN$                 |         sNaN          |
| $\color{#FF0000}{0}\color{#007F00}{11111}\color{#0000FF}{11}$                   | $NaN$                 |         qNaN          |
| $\color{#FF0000}{0}\color{#007F00}{11111}\color{#0000FF}{11}$                         | $NaN$                 | other alternative NaN |
| $\color{#FF0000}{0}\color{#007F00}{01101}\color{#0000FF}{01}$                 | $\frac{1}{3}$        |                       |
