# Activation Layer

+ Simple example
+ type & alpha & beta

---

## Simple example

+ Refer to SimpleExample.py

+ USe ReLU activation function on the input tenosr.

---

## type & alpha & beta

+ Refer to Type+Alpha+Beta.py

+ Adjust type and parameters of the activation function after constructor.

+ Avialable activation function.

| trt.ActivationType |         Original Name         |                          Expression                          |
| :----------------: | :---------------------------: | :----------------------------------------------------------: |
|        CLIP        |        Clip activation        | $f\left(x\right) = \max\left(alpha, \min\left(beta,x\right)\right)$ |
|        ELU         |        Elu activation         | $f\left(x\right) = \left\{\begin{aligned} x \ \ \left(x \ge 0 \right) \\ alpha *\left( \exp\left(x\right)-1\right) \ \left(x \lt 0 \right) \end{aligned}\right.$ |
|    HARD_SIGMOID    |    Hard sigmoid activation    | $f\left(x\right) = \max\left(0,\min\left(1, alpha * x + beta\right)\right)$ |
|     LEAKY_RELU     |     Leaky Relu activation     | $f\left(x\right) = \left\{\begin{aligned} x \ \left(x \ge 0 \right) \\ alpha * x \ \left(x \lt 0 \right) \end{aligned}\right.$ |
|        RELU        |  Rectified Linear activation  |           $f\left(x\right) = \max\left(0,x\right)$           |
|    SCALED_TANH     |    Scaled Tanh activation     |     $f\left(x\right) = alpha*\tanh\left(beta*x \right)$      |
|        SELU        |        Selu activation        | $f\left(x\right) = \left\{\begin{aligned} beta *x \ \ \left(x \ge 0 \right) \\ beta*alpha * \left( \exp\left(x\right)-1\right) \ \left(x \lt 0 \right) \end{aligned}\right.$ |
|      SIGMOID       |      Sigmoid activation       |     $f\left(x\right) = \frac{x}{1 + exp\left(-x\right)}$     |
|      SOFTPLUS      |      Softplus activation      | $f\left(x\right) = alpha *\log\left(\exp\left(beta*x\right) + 1\right)$ |
|      SOFTSIGN      |      Softsign activation      |       $f\left(x\right) = \frac{x}{1 + \left|x\right|}$       |
|        TANH        | Hyperbolic Tangent activation |           $f\left(x\right) = \tanh\left(x\right)$            |
|  THRESHOLDED_RELU  |  Thresholded Relu activation  | $f\left(x\right) = \left\{\begin{aligned} x \ \left(x \gt alpha \right) \\ 0 \ \left(x \le alpha\right) \end{aligned}\right.$ |
