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
|     SIGMOID      |     $f\left(x\right) = \frac{x}{1 + exp\left(-x\right)}$     |
|     SOFTPLUS     | $f\left(x\right) = alpha * \log\left(\exp\left(beta * x\right) + 1\right)$ |
|     SOFTSIGN     |       $f\left(x\right) = \frac{x}{1 + \left|x\right|}$       |
|       TANH       |           $f\left(x\right) = \tanh\left(x\right)$            |
| THRESHOLDED_RELU | $f\left(x\right) = \left\{\begin{aligned} x \ \left(x \gt alpha \right) \\ 0 \ \left(x \le alpha\right) \end{aligned}\right.$ |

+ Default values of parameters

| Name  | Comment |
| :---: | :-----: |
| alpha |    0    |
| beta  |    0    |
