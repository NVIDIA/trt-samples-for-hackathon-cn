# FP8e8m0 - not IEEE754

+ Sign:     $s$ ($p=0$ bit)
+ Exponent: $e$ ($q=8$ bits)
+ Mantissa: $m$ ($r=0$ bits)

+ Normal value ($00000001_2 \le e_2 \le 11111110_2$) (subscript 2 represents the base)

$$
\begin{equation}
\begin{aligned}
E &= e - \left( 2^{q-1} - 1 \right) = e - 127 \\
M &= m \cdot 2^{-r} = m \cdot 2^{0} \\
value &= \left( -1 \right) ^ {s} 2 ^ {E} \left( 1 + M \right) = \left( -1 \right) ^ {s} 2 ^ {-127} 2 ^ {e} \left( m  + 2^{0} \right)
\end{aligned}
\end{equation}
$$

+ Subnormal value ($e_2 = 00000000_2$)

$$
\begin{equation}
\begin{aligned}
E &= 2 - 2^{q-1} = -126 \\
M &= m \cdot 2^{-r} = m \cdot 2^{0} \\
value &= \left( -1 \right) ^ {s} 2 ^ {E} M = \left( -1 \right) ^ {s} 2^{-126} m
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
| all 1 | all 1     | NaN             |

+ Examples

| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponent}\color{#0000FF}{Mantissa}$) | value          | comment          |
| :-:        | :-:            | :-:              |
| None         | $0$               | Zero (not exist)       |
| $\color{#FF0000}{}\color{#007F00}{11111111}\color{#0000FF}{}$     | $3.4028\times10^{+38}$  | Maximum  |
| $\color{#FF0000}{}\color{#007F00}{00000000}\color{#0000FF}{}$     | $-3.4028\times10^{+38}$ | Minimum |
| $\color{#FF0000}{}\color{#007F00}{11111111}\color{#0000FF}{}$         | $NaN$              | NaN     |
| $\color{#FF0000}{}\color{#007F00}{01111101}\color{#0000FF}{}$     | $2.5000\times10^{-01}$ | $\frac{1}{3}$      |

+ All non-negative values

| Number($\color{#FF0000}{Sign}\color{#007F00}{Exponen}\color{#0000FF}{Mantissa}$) | value |
| :-:        | :-:   |
| $\color{#FF0000}{}\color{#007F00}{00000000}\color{#0000FF}{}$     | 5.877e-39  |
| $\color{#FF0000}{}\color{#007F00}{00000001}\color{#0000FF}{}$     | 1.175e-38  |
| $\color{#FF0000}{}\color{#007F00}{00000010}\color{#0000FF}{}$     | 2.351e-38  |
| $\color{#FF0000}{}\color{#007F00}{00000011}\color{#0000FF}{}$     | 4.702e-38  |
| $\color{#FF0000}{}\color{#007F00}{00000100}\color{#0000FF}{}$     | 9.404e-38  |
| $\color{#FF0000}{}\color{#007F00}{00000101}\color{#0000FF}{}$     | 1.881e-37  |
| $\color{#FF0000}{}\color{#007F00}{00000110}\color{#0000FF}{}$     | 3.762e-37  |
| $\color{#FF0000}{}\color{#007F00}{00000111}\color{#0000FF}{}$     | 7.523e-37  |
| $\color{#FF0000}{}\color{#007F00}{00001000}\color{#0000FF}{}$     | 1.505e-36  |
| $\color{#FF0000}{}\color{#007F00}{00001001}\color{#0000FF}{}$     | 3.009e-36  |
| $\color{#FF0000}{}\color{#007F00}{00001010}\color{#0000FF}{}$     | 6.019e-36  |
| $\color{#FF0000}{}\color{#007F00}{00001011}\color{#0000FF}{}$     | 1.204e-35  |
| $\color{#FF0000}{}\color{#007F00}{00001100}\color{#0000FF}{}$     | 2.407e-35  |
| $\color{#FF0000}{}\color{#007F00}{00001101}\color{#0000FF}{}$     | 4.815e-35  |
| $\color{#FF0000}{}\color{#007F00}{00001110}\color{#0000FF}{}$     | 9.630e-35  |
| $\color{#FF0000}{}\color{#007F00}{00001111}\color{#0000FF}{}$     | 1.926e-34  |
| $\color{#FF0000}{}\color{#007F00}{00010000}\color{#0000FF}{}$     | 3.852e-34  |
| $\color{#FF0000}{}\color{#007F00}{00010001}\color{#0000FF}{}$     | 7.704e-34  |
| $\color{#FF0000}{}\color{#007F00}{00010010}\color{#0000FF}{}$     | 1.541e-33  |
| $\color{#FF0000}{}\color{#007F00}{00010011}\color{#0000FF}{}$     | 3.081e-33  |
| $\color{#FF0000}{}\color{#007F00}{00010100}\color{#0000FF}{}$     | 6.163e-33  |
| $\color{#FF0000}{}\color{#007F00}{00010101}\color{#0000FF}{}$     | 1.233e-32  |
| $\color{#FF0000}{}\color{#007F00}{00010110}\color{#0000FF}{}$     | 2.465e-32  |
| $\color{#FF0000}{}\color{#007F00}{00010111}\color{#0000FF}{}$     | 4.930e-32  |
| $\color{#FF0000}{}\color{#007F00}{00011000}\color{#0000FF}{}$     | 9.861e-32  |
| $\color{#FF0000}{}\color{#007F00}{00011001}\color{#0000FF}{}$     | 1.972e-31  |
| $\color{#FF0000}{}\color{#007F00}{00011010}\color{#0000FF}{}$     | 3.944e-31  |
| $\color{#FF0000}{}\color{#007F00}{00011011}\color{#0000FF}{}$     | 7.889e-31  |
| $\color{#FF0000}{}\color{#007F00}{00011100}\color{#0000FF}{}$     | 1.578e-30  |
| $\color{#FF0000}{}\color{#007F00}{00011101}\color{#0000FF}{}$     | 3.155e-30  |
| $\color{#FF0000}{}\color{#007F00}{00011110}\color{#0000FF}{}$     | 6.311e-30  |
| $\color{#FF0000}{}\color{#007F00}{00011111}\color{#0000FF}{}$     | 1.262e-29  |
| $\color{#FF0000}{}\color{#007F00}{00100000}\color{#0000FF}{}$     | 2.524e-29  |
| $\color{#FF0000}{}\color{#007F00}{00100001}\color{#0000FF}{}$     | 5.049e-29  |
| $\color{#FF0000}{}\color{#007F00}{00100010}\color{#0000FF}{}$     | 1.010e-28  |
| $\color{#FF0000}{}\color{#007F00}{00100011}\color{#0000FF}{}$     | 2.019e-28  |
| $\color{#FF0000}{}\color{#007F00}{00100100}\color{#0000FF}{}$     | 4.039e-28  |
| $\color{#FF0000}{}\color{#007F00}{00100101}\color{#0000FF}{}$     | 8.078e-28  |
| $\color{#FF0000}{}\color{#007F00}{00100110}\color{#0000FF}{}$     | 1.616e-27  |
| $\color{#FF0000}{}\color{#007F00}{00100111}\color{#0000FF}{}$     | 3.231e-27  |
| $\color{#FF0000}{}\color{#007F00}{00101000}\color{#0000FF}{}$     | 6.462e-27  |
| $\color{#FF0000}{}\color{#007F00}{00101001}\color{#0000FF}{}$     | 1.292e-26  |
| $\color{#FF0000}{}\color{#007F00}{00101010}\color{#0000FF}{}$     | 2.585e-26  |
| $\color{#FF0000}{}\color{#007F00}{00101011}\color{#0000FF}{}$     | 5.170e-26  |
| $\color{#FF0000}{}\color{#007F00}{00101100}\color{#0000FF}{}$     | 1.034e-25  |
| $\color{#FF0000}{}\color{#007F00}{00101101}\color{#0000FF}{}$     | 2.068e-25  |
| $\color{#FF0000}{}\color{#007F00}{00101110}\color{#0000FF}{}$     | 4.136e-25  |
| $\color{#FF0000}{}\color{#007F00}{00101111}\color{#0000FF}{}$     | 8.272e-25  |
| $\color{#FF0000}{}\color{#007F00}{00110000}\color{#0000FF}{}$     | 1.654e-24  |
| $\color{#FF0000}{}\color{#007F00}{00110001}\color{#0000FF}{}$     | 3.309e-24  |
| $\color{#FF0000}{}\color{#007F00}{00110010}\color{#0000FF}{}$     | 6.617e-24  |
| $\color{#FF0000}{}\color{#007F00}{00110011}\color{#0000FF}{}$     | 1.323e-23  |
| $\color{#FF0000}{}\color{#007F00}{00110100}\color{#0000FF}{}$     | 2.647e-23  |
| $\color{#FF0000}{}\color{#007F00}{00110101}\color{#0000FF}{}$     | 5.294e-23  |
| $\color{#FF0000}{}\color{#007F00}{00110110}\color{#0000FF}{}$     | 1.059e-22  |
| $\color{#FF0000}{}\color{#007F00}{00110111}\color{#0000FF}{}$     | 2.118e-22  |
| $\color{#FF0000}{}\color{#007F00}{00111000}\color{#0000FF}{}$     | 4.235e-22  |
| $\color{#FF0000}{}\color{#007F00}{00111001}\color{#0000FF}{}$     | 8.470e-22  |
| $\color{#FF0000}{}\color{#007F00}{00111010}\color{#0000FF}{}$     | 1.694e-21  |
| $\color{#FF0000}{}\color{#007F00}{00111011}\color{#0000FF}{}$     | 3.388e-21  |
| $\color{#FF0000}{}\color{#007F00}{00111100}\color{#0000FF}{}$     | 6.776e-21  |
| $\color{#FF0000}{}\color{#007F00}{00111101}\color{#0000FF}{}$     | 1.355e-20  |
| $\color{#FF0000}{}\color{#007F00}{00111110}\color{#0000FF}{}$     | 2.711e-20  |
| $\color{#FF0000}{}\color{#007F00}{00111111}\color{#0000FF}{}$     | 5.421e-20  |
| $\color{#FF0000}{}\color{#007F00}{01000000}\color{#0000FF}{}$     | 1.084e-19  |
| $\color{#FF0000}{}\color{#007F00}{01000001}\color{#0000FF}{}$     | 2.168e-19  |
| $\color{#FF0000}{}\color{#007F00}{01000010}\color{#0000FF}{}$     | 4.337e-19  |
| $\color{#FF0000}{}\color{#007F00}{01000011}\color{#0000FF}{}$     | 8.674e-19  |
| $\color{#FF0000}{}\color{#007F00}{01000100}\color{#0000FF}{}$     | 1.735e-18  |
| $\color{#FF0000}{}\color{#007F00}{01000101}\color{#0000FF}{}$     | 3.469e-18  |
| $\color{#FF0000}{}\color{#007F00}{01000110}\color{#0000FF}{}$     | 6.939e-18  |
| $\color{#FF0000}{}\color{#007F00}{01000111}\color{#0000FF}{}$     | 1.388e-17  |
| $\color{#FF0000}{}\color{#007F00}{01001000}\color{#0000FF}{}$     | 2.776e-17  |
| $\color{#FF0000}{}\color{#007F00}{01001001}\color{#0000FF}{}$     | 5.551e-17  |
| $\color{#FF0000}{}\color{#007F00}{01001010}\color{#0000FF}{}$     | 1.110e-16  |
| $\color{#FF0000}{}\color{#007F00}{01001011}\color{#0000FF}{}$     | 2.220e-16  |
| $\color{#FF0000}{}\color{#007F00}{01001100}\color{#0000FF}{}$     | 4.441e-16  |
| $\color{#FF0000}{}\color{#007F00}{01001101}\color{#0000FF}{}$     | 8.882e-16  |
| $\color{#FF0000}{}\color{#007F00}{01001110}\color{#0000FF}{}$     | 1.776e-15  |
| $\color{#FF0000}{}\color{#007F00}{01001111}\color{#0000FF}{}$     | 3.553e-15  |
| $\color{#FF0000}{}\color{#007F00}{01010000}\color{#0000FF}{}$     | 7.105e-15  |
| $\color{#FF0000}{}\color{#007F00}{01010001}\color{#0000FF}{}$     | 1.421e-14  |
| $\color{#FF0000}{}\color{#007F00}{01010010}\color{#0000FF}{}$     | 2.842e-14  |
| $\color{#FF0000}{}\color{#007F00}{01010011}\color{#0000FF}{}$     | 5.684e-14  |
| $\color{#FF0000}{}\color{#007F00}{01010100}\color{#0000FF}{}$     | 1.137e-13  |
| $\color{#FF0000}{}\color{#007F00}{01010101}\color{#0000FF}{}$     | 2.274e-13  |
| $\color{#FF0000}{}\color{#007F00}{01010110}\color{#0000FF}{}$     | 4.547e-13  |
| $\color{#FF0000}{}\color{#007F00}{01010111}\color{#0000FF}{}$     | 9.095e-13  |
| $\color{#FF0000}{}\color{#007F00}{01011000}\color{#0000FF}{}$     | 1.819e-12  |
| $\color{#FF0000}{}\color{#007F00}{01011001}\color{#0000FF}{}$     | 3.638e-12  |
| $\color{#FF0000}{}\color{#007F00}{01011010}\color{#0000FF}{}$     | 7.276e-12  |
| $\color{#FF0000}{}\color{#007F00}{01011011}\color{#0000FF}{}$     | 1.455e-11  |
| $\color{#FF0000}{}\color{#007F00}{01011100}\color{#0000FF}{}$     | 2.910e-11  |
| $\color{#FF0000}{}\color{#007F00}{01011101}\color{#0000FF}{}$     | 5.821e-11  |
| $\color{#FF0000}{}\color{#007F00}{01011110}\color{#0000FF}{}$     | 1.164e-10  |
| $\color{#FF0000}{}\color{#007F00}{01011111}\color{#0000FF}{}$     | 2.328e-10  |
| $\color{#FF0000}{}\color{#007F00}{01100000}\color{#0000FF}{}$     | 4.657e-10  |
| $\color{#FF0000}{}\color{#007F00}{01100001}\color{#0000FF}{}$     | 9.313e-10  |
| $\color{#FF0000}{}\color{#007F00}{01100010}\color{#0000FF}{}$     | 1.863e-09  |
| $\color{#FF0000}{}\color{#007F00}{01100011}\color{#0000FF}{}$     | 3.725e-09  |
| $\color{#FF0000}{}\color{#007F00}{01100100}\color{#0000FF}{}$     | 7.451e-09  |
| $\color{#FF0000}{}\color{#007F00}{01100101}\color{#0000FF}{}$     | 1.490e-08  |
| $\color{#FF0000}{}\color{#007F00}{01100110}\color{#0000FF}{}$     | 2.980e-08  |
| $\color{#FF0000}{}\color{#007F00}{01100111}\color{#0000FF}{}$     | 5.960e-08  |
| $\color{#FF0000}{}\color{#007F00}{01101000}\color{#0000FF}{}$     | 1.192e-07  |
| $\color{#FF0000}{}\color{#007F00}{01101001}\color{#0000FF}{}$     | 2.384e-07  |
| $\color{#FF0000}{}\color{#007F00}{01101010}\color{#0000FF}{}$     | 4.768e-07  |
| $\color{#FF0000}{}\color{#007F00}{01101011}\color{#0000FF}{}$     | 9.537e-07  |
| $\color{#FF0000}{}\color{#007F00}{01101100}\color{#0000FF}{}$     | 1.907e-06  |
| $\color{#FF0000}{}\color{#007F00}{01101101}\color{#0000FF}{}$     | 3.815e-06  |
| $\color{#FF0000}{}\color{#007F00}{01101110}\color{#0000FF}{}$     | 7.629e-06  |
| $\color{#FF0000}{}\color{#007F00}{01101111}\color{#0000FF}{}$     | 1.526e-05  |
| $\color{#FF0000}{}\color{#007F00}{01110000}\color{#0000FF}{}$     | 3.052e-05  |
| $\color{#FF0000}{}\color{#007F00}{01110001}\color{#0000FF}{}$     | 6.104e-05  |
| $\color{#FF0000}{}\color{#007F00}{01110010}\color{#0000FF}{}$     | 1.221e-04  |
| $\color{#FF0000}{}\color{#007F00}{01110011}\color{#0000FF}{}$     | 2.441e-04  |
| $\color{#FF0000}{}\color{#007F00}{01110100}\color{#0000FF}{}$     | 4.883e-04  |
| $\color{#FF0000}{}\color{#007F00}{01110101}\color{#0000FF}{}$     | 9.766e-04  |
| $\color{#FF0000}{}\color{#007F00}{01110110}\color{#0000FF}{}$     | 1.953e-03  |
| $\color{#FF0000}{}\color{#007F00}{01110111}\color{#0000FF}{}$     | 3.906e-03  |
| $\color{#FF0000}{}\color{#007F00}{01111000}\color{#0000FF}{}$     | 7.812e-03  |
| $\color{#FF0000}{}\color{#007F00}{01111001}\color{#0000FF}{}$     | 1.562e-02  |
| $\color{#FF0000}{}\color{#007F00}{01111010}\color{#0000FF}{}$     | 3.125e-02  |
| $\color{#FF0000}{}\color{#007F00}{01111011}\color{#0000FF}{}$     | 6.250e-02  |
| $\color{#FF0000}{}\color{#007F00}{01111100}\color{#0000FF}{}$     | 1.250e-01  |
| $\color{#FF0000}{}\color{#007F00}{01111101}\color{#0000FF}{}$     | 2.500e-01  |
| $\color{#FF0000}{}\color{#007F00}{01111110}\color{#0000FF}{}$     | 5.000e-01  |
| $\color{#FF0000}{}\color{#007F00}{01111111}\color{#0000FF}{}$     | 1.000e+00  |
| $\color{#FF0000}{}\color{#007F00}{10000000}\color{#0000FF}{}$     | 2.000e+00  |
| $\color{#FF0000}{}\color{#007F00}{10000001}\color{#0000FF}{}$     | 4.000e+00  |
| $\color{#FF0000}{}\color{#007F00}{10000010}\color{#0000FF}{}$     | 8.000e+00  |
| $\color{#FF0000}{}\color{#007F00}{10000011}\color{#0000FF}{}$     | 1.600e+01  |
| $\color{#FF0000}{}\color{#007F00}{10000100}\color{#0000FF}{}$     | 3.200e+01  |
| $\color{#FF0000}{}\color{#007F00}{10000101}\color{#0000FF}{}$     | 6.400e+01  |
| $\color{#FF0000}{}\color{#007F00}{10000110}\color{#0000FF}{}$     | 1.280e+02  |
| $\color{#FF0000}{}\color{#007F00}{10000111}\color{#0000FF}{}$     | 2.560e+02  |
| $\color{#FF0000}{}\color{#007F00}{10001000}\color{#0000FF}{}$     | 5.120e+02  |
| $\color{#FF0000}{}\color{#007F00}{10001001}\color{#0000FF}{}$     | 1.024e+03  |
| $\color{#FF0000}{}\color{#007F00}{10001010}\color{#0000FF}{}$     | 2.048e+03  |
| $\color{#FF0000}{}\color{#007F00}{10001011}\color{#0000FF}{}$     | 4.096e+03  |
| $\color{#FF0000}{}\color{#007F00}{10001100}\color{#0000FF}{}$     | 8.192e+03  |
| $\color{#FF0000}{}\color{#007F00}{10001101}\color{#0000FF}{}$     | 1.638e+04  |
| $\color{#FF0000}{}\color{#007F00}{10001110}\color{#0000FF}{}$     | 3.277e+04  |
| $\color{#FF0000}{}\color{#007F00}{10001111}\color{#0000FF}{}$     | 6.554e+04  |
| $\color{#FF0000}{}\color{#007F00}{10010000}\color{#0000FF}{}$     | 1.311e+05  |
| $\color{#FF0000}{}\color{#007F00}{10010001}\color{#0000FF}{}$     | 2.621e+05  |
| $\color{#FF0000}{}\color{#007F00}{10010010}\color{#0000FF}{}$     | 5.243e+05  |
| $\color{#FF0000}{}\color{#007F00}{10010011}\color{#0000FF}{}$     | 1.049e+06  |
| $\color{#FF0000}{}\color{#007F00}{10010100}\color{#0000FF}{}$     | 2.097e+06  |
| $\color{#FF0000}{}\color{#007F00}{10010101}\color{#0000FF}{}$     | 4.194e+06  |
| $\color{#FF0000}{}\color{#007F00}{10010110}\color{#0000FF}{}$     | 8.389e+06  |
| $\color{#FF0000}{}\color{#007F00}{10010111}\color{#0000FF}{}$     | 1.678e+07  |
| $\color{#FF0000}{}\color{#007F00}{10011000}\color{#0000FF}{}$     | 3.355e+07  |
| $\color{#FF0000}{}\color{#007F00}{10011001}\color{#0000FF}{}$     | 6.711e+07  |
| $\color{#FF0000}{}\color{#007F00}{10011010}\color{#0000FF}{}$     | 1.342e+08  |
| $\color{#FF0000}{}\color{#007F00}{10011011}\color{#0000FF}{}$     | 2.684e+08  |
| $\color{#FF0000}{}\color{#007F00}{10011100}\color{#0000FF}{}$     | 5.369e+08  |
| $\color{#FF0000}{}\color{#007F00}{10011101}\color{#0000FF}{}$     | 1.074e+09  |
| $\color{#FF0000}{}\color{#007F00}{10011110}\color{#0000FF}{}$     | 2.147e+09  |
| $\color{#FF0000}{}\color{#007F00}{10011111}\color{#0000FF}{}$     | 4.295e+09  |
| $\color{#FF0000}{}\color{#007F00}{10100000}\color{#0000FF}{}$     | 8.590e+09  |
| $\color{#FF0000}{}\color{#007F00}{10100001}\color{#0000FF}{}$     | 1.718e+10  |
| $\color{#FF0000}{}\color{#007F00}{10100010}\color{#0000FF}{}$     | 3.436e+10  |
| $\color{#FF0000}{}\color{#007F00}{10100011}\color{#0000FF}{}$     | 6.872e+10  |
| $\color{#FF0000}{}\color{#007F00}{10100100}\color{#0000FF}{}$     | 1.374e+11  |
| $\color{#FF0000}{}\color{#007F00}{10100101}\color{#0000FF}{}$     | 2.749e+11  |
| $\color{#FF0000}{}\color{#007F00}{10100110}\color{#0000FF}{}$     | 5.498e+11  |
| $\color{#FF0000}{}\color{#007F00}{10100111}\color{#0000FF}{}$     | 1.100e+12  |
| $\color{#FF0000}{}\color{#007F00}{10101000}\color{#0000FF}{}$     | 2.199e+12  |
| $\color{#FF0000}{}\color{#007F00}{10101001}\color{#0000FF}{}$     | 4.398e+12  |
| $\color{#FF0000}{}\color{#007F00}{10101010}\color{#0000FF}{}$     | 8.796e+12  |
| $\color{#FF0000}{}\color{#007F00}{10101011}\color{#0000FF}{}$     | 1.759e+13  |
| $\color{#FF0000}{}\color{#007F00}{10101100}\color{#0000FF}{}$     | 3.518e+13  |
| $\color{#FF0000}{}\color{#007F00}{10101101}\color{#0000FF}{}$     | 7.037e+13  |
| $\color{#FF0000}{}\color{#007F00}{10101110}\color{#0000FF}{}$     | 1.407e+14  |
| $\color{#FF0000}{}\color{#007F00}{10101111}\color{#0000FF}{}$     | 2.815e+14  |
| $\color{#FF0000}{}\color{#007F00}{10110000}\color{#0000FF}{}$     | 5.629e+14  |
| $\color{#FF0000}{}\color{#007F00}{10110001}\color{#0000FF}{}$     | 1.126e+15  |
| $\color{#FF0000}{}\color{#007F00}{10110010}\color{#0000FF}{}$     | 2.252e+15  |
| $\color{#FF0000}{}\color{#007F00}{10110011}\color{#0000FF}{}$     | 4.504e+15  |
| $\color{#FF0000}{}\color{#007F00}{10110100}\color{#0000FF}{}$     | 9.007e+15  |
| $\color{#FF0000}{}\color{#007F00}{10110101}\color{#0000FF}{}$     | 1.801e+16  |
| $\color{#FF0000}{}\color{#007F00}{10110110}\color{#0000FF}{}$     | 3.603e+16  |
| $\color{#FF0000}{}\color{#007F00}{10110111}\color{#0000FF}{}$     | 7.206e+16  |
| $\color{#FF0000}{}\color{#007F00}{10111000}\color{#0000FF}{}$     | 1.441e+17  |
| $\color{#FF0000}{}\color{#007F00}{10111001}\color{#0000FF}{}$     | 2.882e+17  |
| $\color{#FF0000}{}\color{#007F00}{10111010}\color{#0000FF}{}$     | 5.765e+17  |
| $\color{#FF0000}{}\color{#007F00}{10111011}\color{#0000FF}{}$     | 1.153e+18  |
| $\color{#FF0000}{}\color{#007F00}{10111100}\color{#0000FF}{}$     | 2.306e+18  |
| $\color{#FF0000}{}\color{#007F00}{10111101}\color{#0000FF}{}$     | 4.612e+18  |
| $\color{#FF0000}{}\color{#007F00}{10111110}\color{#0000FF}{}$     | 9.223e+18  |
| $\color{#FF0000}{}\color{#007F00}{10111111}\color{#0000FF}{}$     | 1.845e+19  |
| $\color{#FF0000}{}\color{#007F00}{11000000}\color{#0000FF}{}$     | 3.689e+19  |
| $\color{#FF0000}{}\color{#007F00}{11000001}\color{#0000FF}{}$     | 7.379e+19  |
| $\color{#FF0000}{}\color{#007F00}{11000010}\color{#0000FF}{}$     | 1.476e+20  |
| $\color{#FF0000}{}\color{#007F00}{11000011}\color{#0000FF}{}$     | 2.951e+20  |
| $\color{#FF0000}{}\color{#007F00}{11000100}\color{#0000FF}{}$     | 5.903e+20  |
| $\color{#FF0000}{}\color{#007F00}{11000101}\color{#0000FF}{}$     | 1.181e+21  |
| $\color{#FF0000}{}\color{#007F00}{11000110}\color{#0000FF}{}$     | 2.361e+21  |
| $\color{#FF0000}{}\color{#007F00}{11000111}\color{#0000FF}{}$     | 4.722e+21  |
| $\color{#FF0000}{}\color{#007F00}{11001000}\color{#0000FF}{}$     | 9.445e+21  |
| $\color{#FF0000}{}\color{#007F00}{11001001}\color{#0000FF}{}$     | 1.889e+22  |
| $\color{#FF0000}{}\color{#007F00}{11001010}\color{#0000FF}{}$     | 3.778e+22  |
| $\color{#FF0000}{}\color{#007F00}{11001011}\color{#0000FF}{}$     | 7.556e+22  |
| $\color{#FF0000}{}\color{#007F00}{11001100}\color{#0000FF}{}$     | 1.511e+23  |
| $\color{#FF0000}{}\color{#007F00}{11001101}\color{#0000FF}{}$     | 3.022e+23  |
| $\color{#FF0000}{}\color{#007F00}{11001110}\color{#0000FF}{}$     | 6.045e+23  |
| $\color{#FF0000}{}\color{#007F00}{11001111}\color{#0000FF}{}$     | 1.209e+24  |
| $\color{#FF0000}{}\color{#007F00}{11010000}\color{#0000FF}{}$     | 2.418e+24  |
| $\color{#FF0000}{}\color{#007F00}{11010001}\color{#0000FF}{}$     | 4.836e+24  |
| $\color{#FF0000}{}\color{#007F00}{11010010}\color{#0000FF}{}$     | 9.671e+24  |
| $\color{#FF0000}{}\color{#007F00}{11010011}\color{#0000FF}{}$     | 1.934e+25  |
| $\color{#FF0000}{}\color{#007F00}{11010100}\color{#0000FF}{}$     | 3.869e+25  |
| $\color{#FF0000}{}\color{#007F00}{11010101}\color{#0000FF}{}$     | 7.737e+25  |
| $\color{#FF0000}{}\color{#007F00}{11010110}\color{#0000FF}{}$     | 1.547e+26  |
| $\color{#FF0000}{}\color{#007F00}{11010111}\color{#0000FF}{}$     | 3.095e+26  |
| $\color{#FF0000}{}\color{#007F00}{11011000}\color{#0000FF}{}$     | 6.190e+26  |
| $\color{#FF0000}{}\color{#007F00}{11011001}\color{#0000FF}{}$     | 1.238e+27  |
| $\color{#FF0000}{}\color{#007F00}{11011010}\color{#0000FF}{}$     | 2.476e+27  |
| $\color{#FF0000}{}\color{#007F00}{11011011}\color{#0000FF}{}$     | 4.952e+27  |
| $\color{#FF0000}{}\color{#007F00}{11011100}\color{#0000FF}{}$     | 9.904e+27  |
| $\color{#FF0000}{}\color{#007F00}{11011101}\color{#0000FF}{}$     | 1.981e+28  |
| $\color{#FF0000}{}\color{#007F00}{11011110}\color{#0000FF}{}$     | 3.961e+28  |
| $\color{#FF0000}{}\color{#007F00}{11011111}\color{#0000FF}{}$     | 7.923e+28  |
| $\color{#FF0000}{}\color{#007F00}{11100000}\color{#0000FF}{}$     | 1.585e+29  |
| $\color{#FF0000}{}\color{#007F00}{11100001}\color{#0000FF}{}$     | 3.169e+29  |
| $\color{#FF0000}{}\color{#007F00}{11100010}\color{#0000FF}{}$     | 6.338e+29  |
| $\color{#FF0000}{}\color{#007F00}{11100011}\color{#0000FF}{}$     | 1.268e+30  |
| $\color{#FF0000}{}\color{#007F00}{11100100}\color{#0000FF}{}$     | 2.535e+30  |
| $\color{#FF0000}{}\color{#007F00}{11100101}\color{#0000FF}{}$     | 5.071e+30  |
| $\color{#FF0000}{}\color{#007F00}{11100110}\color{#0000FF}{}$     | 1.014e+31  |
| $\color{#FF0000}{}\color{#007F00}{11100111}\color{#0000FF}{}$     | 2.028e+31  |
| $\color{#FF0000}{}\color{#007F00}{11101000}\color{#0000FF}{}$     | 4.056e+31  |
| $\color{#FF0000}{}\color{#007F00}{11101001}\color{#0000FF}{}$     | 8.113e+31  |
| $\color{#FF0000}{}\color{#007F00}{11101010}\color{#0000FF}{}$     | 1.623e+32  |
| $\color{#FF0000}{}\color{#007F00}{11101011}\color{#0000FF}{}$     | 3.245e+32  |
| $\color{#FF0000}{}\color{#007F00}{11101100}\color{#0000FF}{}$     | 6.490e+32  |
| $\color{#FF0000}{}\color{#007F00}{11101101}\color{#0000FF}{}$     | 1.298e+33  |
| $\color{#FF0000}{}\color{#007F00}{11101110}\color{#0000FF}{}$     | 2.596e+33  |
| $\color{#FF0000}{}\color{#007F00}{11101111}\color{#0000FF}{}$     | 5.192e+33  |
| $\color{#FF0000}{}\color{#007F00}{11110000}\color{#0000FF}{}$     | 1.038e+34  |
| $\color{#FF0000}{}\color{#007F00}{11110001}\color{#0000FF}{}$     | 2.077e+34  |
| $\color{#FF0000}{}\color{#007F00}{11110010}\color{#0000FF}{}$     | 4.154e+34  |
| $\color{#FF0000}{}\color{#007F00}{11110011}\color{#0000FF}{}$     | 8.308e+34  |
| $\color{#FF0000}{}\color{#007F00}{11110100}\color{#0000FF}{}$     | 1.662e+35  |
| $\color{#FF0000}{}\color{#007F00}{11110101}\color{#0000FF}{}$     | 3.323e+35  |
| $\color{#FF0000}{}\color{#007F00}{11110110}\color{#0000FF}{}$     | 6.646e+35  |
| $\color{#FF0000}{}\color{#007F00}{11110111}\color{#0000FF}{}$     | 1.329e+36  |
| $\color{#FF0000}{}\color{#007F00}{11111000}\color{#0000FF}{}$     | 2.658e+36  |
| $\color{#FF0000}{}\color{#007F00}{11111001}\color{#0000FF}{}$     | 5.317e+36  |
| $\color{#FF0000}{}\color{#007F00}{11111010}\color{#0000FF}{}$     | 1.063e+37  |
| $\color{#FF0000}{}\color{#007F00}{11111011}\color{#0000FF}{}$     | 2.127e+37  |
| $\color{#FF0000}{}\color{#007F00}{11111100}\color{#0000FF}{}$     | 4.254e+37  |
| $\color{#FF0000}{}\color{#007F00}{11111101}\color{#0000FF}{}$     | 8.507e+37  |
| $\color{#FF0000}{}\color{#007F00}{11111110}\color{#0000FF}{}$     | 1.701e+38  |
| $\color{#FF0000}{}\color{#007F00}{11111111}\color{#0000FF}{}$     | nan  |
