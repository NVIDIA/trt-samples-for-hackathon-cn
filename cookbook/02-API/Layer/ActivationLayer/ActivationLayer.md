# Activation Layer

+ Common
+ Simple example
+ type & alpha & beta

---

## Common

+ Input tensor
  + T0

+ Output tensor
  + T1

+ Data type
  + T0, T1: float32, float16, int8

+ Shape
  + T1.shape == T0.shape

+ Attribution and default value
  + alpha = 0, parameter for some find of activation function
  + beta = 0, parameter for some find of activation function
  + type, avialable activation function, shown as table below

| trt.ActivationType |   Original Name    |                                                                                  Expression                                                                                  |
| :----------------: | :----------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|        CLIP        |        Clip        |                                                     $f\left(x\right) = \max\left(alpha, \min\left(beta,x\right)\right)$                                                      |
|        ELU         |        Elu         |       $f\left(x\right) = \left\{\begin{aligned} x \ \ \left(x \ge 0 \right) \\ alpha *\left( \exp\left(x\right)-1\right) \ \left(x \lt 0 \right) \end{aligned}\right.$       |
|    HARD_SIGMOID    |    Hard sigmoid    |                                                 $f\left(x\right) = \max\left(0,\min\left(1, alpha * x + beta\right)\right)$                                                  |
|     LEAKY_RELU     |     Leaky Relu     |                        $f\left(x\right) = \left\{\begin{aligned} x \ \left(x \ge 0 \right) \\ alpha * x \ \left(x \lt 0 \right) \end{aligned}\right.$                        |
|        RELU        |  Rectified Linear  |                                                                   $f\left(x\right) = \max\left(0,x\right)$                                                                   |
|    SCALED_TANH     |    Scaled Tanh     |                                                             $f\left(x\right) = alpha*\tanh\left(beta*x \right)$                                                              |
|        SELU        |        Selu        | $f\left(x\right) = \left\{\begin{aligned} beta *x \ \ \left(x \ge 0 \right) \\ beta*alpha * \left( \exp\left(x\right)-1\right) \ \left(x \lt 0 \right) \end{aligned}\right.$ |
|      SIGMOID       |      Sigmoid       |                                                             $f\left(x\right) = \frac{x}{1 + exp\left(-x\right)}$                                                             |
|      SOFTPLUS      |      Softplus      |                                                   $f\left(x\right) = alpha *\log\left(\exp\left(beta*x\right) + 1\right)$                                                    |
|      SOFTSIGN      |      Softsign      |                                                              $f\left(x\right) = \frac{x}{1 + \lvert x\lvert  }$                                                              |
|        TANH        | Hyperbolic Tangent |                                                                   $f\left(x\right) = \tanh\left(x\right)$                                                                    |
|  THRESHOLDED_RELU  |  Thresholded Relu  |                        $f\left(x\right) = \left\{\begin{aligned} x \ \left(x \gt alpha \right) \\ 0 \ \left(x \le alpha\right) \end{aligned}\right.$                         |

---

## Simple example

+ Refer to SimpleExample.py

+ Use ReLU as activation function.

---

## type & alpha & beta

+ Refer to Type+Alpha+Beta.py

+ Reset type and parameters after constructor.
