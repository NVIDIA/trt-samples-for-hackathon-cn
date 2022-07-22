# Activation 层
+ 初始示例代码
+ type & alpha & beta

---
### 初始示例代码
+ 见 SimpleUsage.py

+ 输入张量形状 (1,1,3,3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            -4. & -3. & -2. \\
            -1. &  0. &  1. \\
             1. &  3. &  4.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (1,1,3,3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            0. & 0. & 0. \\
            0. & 0. & 1. \\
            2. & 3. & 4.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### type & alpha & beta
+ 见 Type+Alpha+Beta.py，在构建 Activation 层后再调整其类型和参数

+ 这里设置使用 Clip 激活函数并使输出值限制在 -2 到 2 之间，输出张量形状 (1,1，3,3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            -2. & -2. & -2. \\
            -1. &  0. &  1. \\
             1. &  2. &  2.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 可用的激活函数类型
| trt.ActivationType 名 |             原名              |                            表达式                            |
| :-------------------: | :---------------------------: | :----------------------------------------------------------: |
|         CLIP          |        Clip activation        | $f\left(x\right) = \max\left(alpha, \min\left(beta,x\right)\right)$ |
|          ELU          |        Elu activation         | $f\left(x\right) = \left\{\begin{aligned} x \ \ \left(x \ge 0 \right) \\ alpha * \left( \exp\left(x\right)-1\right) \ \left(x \lt 0 \right) \end{aligned}\right.$ |
|     HARD_SIGMOID      |    Hard sigmoid activation    | $f\left(x\right) = \max\left(0,\min\left(1, alpha * x + beta\right)\right)$ |
|      LEAKY_RELU       |     Leaky Relu activation     | $f\left(x\right) = \left\{\begin{aligned} x \ \left(x \ge 0 \right) \\ alpha * x \ \left(x \lt 0 \right) \end{aligned}\right.$ |
|         RELU          |  Rectified Linear activation  |           $f\left(x\right) = \max\left(0,x\right)$           |
|      SCALED_TANH      |    Scaled Tanh activation     |   $f\left(x\right) = alpha * \tanh\left( beta * x \right)$   |
|         SELU          |        Selu activation        | $f\left(x\right) = \left\{\begin{aligned} beta * x \ \ \left(x \ge 0 \right) \\ beta * alpha * \left( \exp\left(x\right)-1\right) \ \left(x \lt 0 \right) \end{aligned}\right.$ |
|        SIGMOID        |      Sigmoid activation       |     $f\left(x\right) = \frac{x}{1 + exp\left(-x\right)}$     |
|       SOFTPLUS        |      Softplus activation      | $f\left(x\right) = alpha * \log\left(\exp\left(beta * x\right) + 1\right)$ |
|       SOFTSIGN        |      Softsign activation      |       $f\left(x\right) = \frac{x}{1 + \left|x\right|}$       |
|         TANH          | Hyperbolic Tangent activation |           $f\left(x\right) = \tanh\left(x\right)$            |
|   THRESHOLDED_RELU    |  Thresholded Relu activation  | $f\left(x\right) = \left\{\begin{aligned} x \ \left(x \gt alpha \right) \\ 0 \ \left(x \textcolor[rgb]{1,0,0}{\le} alpha\right) \end{aligned}\right.$ |