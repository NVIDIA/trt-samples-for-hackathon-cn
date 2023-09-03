# S1E4M3B0(FP8e4m3) - not IEEE754

+ SignBit ($s$): 1
+ Exponent ($k$): 4
+ Mantissa ($n$): 3
+ Bias ($b$): 0

+ Special value
| Mantissa   | all 0             | not all 0         |
| :-:        | :-:               | :-:               |
| $e = 0000_2$ | Signed Zero       | Subnormal Value   |
| $e = 1111_2$ | Signed Infinity   |       NaN         |

+ Normal value ($0001_2 \le e_2 \le 1110_2$)
$$
\begin{equation}
\begin{aligned}
E &= e_{10} - \left( 2^{k-1} - 1 \right) \\
M &= f_{10} \cdot 2^{-n} \\
value &= \left(-1\right)^{s}2^{E}\left(1+M\right)
\end{aligned}
\end{equation}
$$

+ Subnormal value ($e_2 = 0000_2$)
$$
\begin{equation}
\begin{aligned}
E &= 2 - 2^{k-1} = -6 \\
M &= f_{10} \cdot 2^{-n} \\
value &= \left(-1\right)^{s}2^{E}M
\end{aligned}
\end{equation}
$$

+ Examples
| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponent}\color{#0000FF}{Mantissa}$)  | value                 |        comment        |
| :-:                                                                                | :-:                   | :-:                   |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{000}$                         | $+0$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{001}$                        | $1.9531\times10^{-03}$    |   Minimum subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{0000}\color{#0000FF}{111}$                         | $1.3672\times10^{-02}$    |   Maximum subnormal   |
| $\color{#FF0000}{0}\color{#007F00}{0001}\color{#0000FF}{000}$                        | $1.5625\times10^{-02}$    |    Minimum normal     |
| $\color{#FF0000}{0}\color{#007F00}{0110}\color{#0000FF}{111}$                       |  $1 - 2^{-4}$        |  largest number < 1   |
| $\color{#FF0000}{0}\color{#007F00}{0111}\color{#0000FF}{000}$                        |  $1$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{0111}\color{#0000FF}{001}$                       |  $1 + 2^{-3}$        |  smallest number > 1  |
| $\color{#FF0000}{0}\color{#007F00}{1000}\color{#0000FF}{000}$                        |  $2$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{1000}\color{#0000FF}{100}$                       |  $3$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{000}$                   |  $4$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{010}$             |  $5$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{1001}\color{#0000FF}{100}$             |  $6$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{\bold{1111}}\color{#0000FF}{\bold{110}}$    | $4.4800\times10^{+02}$    |        Maximum        |
| $\color{#FF0000}{1}\color{#007F00}{\bold{1111}}\color{#0000FF}{\bold{110}}$    | $-4.4800\times10^{+02}$    |     Maximum negtive   |
| $\color{#FF0000}{1}\color{#007F00}{0000}\color{#0000FF}{000}$                         | $-0$                  |                       |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{000}$                         | $+\infty$             |   positive infinity   |
| $\color{#FF0000}{1}\color{#007F00}{1111}\color{#0000FF}{000}$                         | $-\infty$             |   negative infinity   |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{001}$                        | $NaN$                 |         sNaN          |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{101}$                   | $NaN$                 |         qNaN          |
| $\color{#FF0000}{0}\color{#007F00}{1111}\color{#0000FF}{111}$                         | $NaN$                 | other alternative NaN |
| $\color{#FF0000}{0}\color{#007F00}{0101}\color{#0000FF}{011}$                 | $\frac{1}{3}$        |                       |
