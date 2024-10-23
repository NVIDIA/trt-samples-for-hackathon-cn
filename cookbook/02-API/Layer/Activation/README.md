# Activation Layer

+ Steps to run.

```bash
python3 main.py
```

+ Alternative values of `trt.ActivationType`

|       Name       |                           Comment                            |
| :--------------: | :----------------------------------------------------------: |
|       CLIP       | $f\left(x\right) = \max\left(alpha, \min\left(beta,x\right)\right)$ |
|       ELU        | $f\left(x\right) = \left\{\begin{aligned} x \ \ \left(x \ge 0 \right) \\ alpha * \left( \exp\left(x\right)-1\right) \ \left(x \lt 0 \right) \end{aligned}\right.$ |
|   HARD_SIGMOID   | $f\left(x\right) = \max\left(0,\min\left(1, alpha * x + beta\right)\right)$ |
|    LEAKY_RELU    | $f\left(x\right) = \left\{\begin{aligned} x \ \left(x \ge 0 \right) \\ alpha * x \ \left(x \lt 0 \right) \end{aligned}\right.$ |
|       RELU       |           $f\left(x\right) = \max\left(0,x\right)$           |
|   SCALED_TANH    |   $f\left(x\right) = alpha * \tanh\left( beta * x \right)$   |
|       SELU       | $f\left(x\right) = \left\{\begin{aligned} beta * x \ \ \left(x \ge 0 \right) \\ beta * alpha * \left( \exp\left(x\right)-1\right) \ \left(x \lt 0 \right) \end{aligned}\right.$ |
|     SIGMOID      |     $f\left(x\right) = \frac{1}{1 + exp\left(-x\right)}$     |
|     SOFTPLUS     | $f\left(x\right) = alpha * \log\left(\exp\left(beta * x\right) + 1\right)$ |
|     SOFTSIGN     |       $f\left(x\right) = \frac{x}{1 + \left|x\right|}$       |
|       TANH       |           $f\left(x\right) = \tanh\left(x\right)$            |
| THRESHOLDED_RELU | $f\left(x\right) = \left\{\begin{aligned} x \ \left(x \gt alpha \right) \\ 0 \ \left(x \le alpha\right) \end{aligned}\right.$ |
| GELU_ERF | $\frac{1}{2}x\left(1 + \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$ |
| GELU_TANH* | $\frac{1}{2}x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$ |

+ *: GELU_TANH is a approximation of GELU_ERF by solving $a$ and $b$ from ([reference](https://www.cvmart.net/community/detail/2063)):

$$
\left\{\begin{matrix}
    \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right) - \tanh\left(ax+bx^3\right) = 0x + o\left(x\right) \\
    \underset{b}{\mathrm{argmin}}\ \underset{x}{\mathrm{argmax}}\left|\mathrm{erf}\left(\frac{x}{\sqrt{2}}\right) - \tanh\left(ax+bx^3\right)\right|
\end{matrix}\right.
$$

+ $b=\frac{4-\pi}{6\pi}$ (parameter 0.044715 becomes 0.045540 in the formula) if solving $a$ and $b$ from:
$$
\mathrm{erf}\left(\frac{x}{\sqrt{2}}\right) - \tanh\left(ax+bx^3\right) = 0x + 0x^3 +o\left(x^3\right)
$$


+ Default values of parameters

| Name  | Comment |
| :---: | :-----: |
| alpha |    0    |
| beta  |    0    |
