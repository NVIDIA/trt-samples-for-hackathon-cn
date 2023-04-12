# Unary Layer

+ Simple example
+ op

---

## Simple example

+ Refer to SimpleExample.py
---

## op

+ Refer to Op.py
+ Adjust content of the one hot layer after constructor.

+ Available unary function

|         trt.UnaryOperation 名          |                              函数                               |    支持的数据类型    |
| :------------------------------------: | :-------------------------------------------------------------: | :------------------: |
|                  ABS                   |                        $\lvert x \rvert$                        |    FP32/FP16/INT8    |
|                  CEIL                  |                        $\lceil x \rceil$                        |    FP32/FP16/INT8    |
|                  ERF                   | $erf \left( x \right) = \int_{0}^{x} \exp\left(-t^{2}\right)dt$ |    FP32/FP16/INT8    |
|                  EXP                   |                     $\exp \left( x \right)$                     |    FP32/FP16/INT8    |
|                 FLOOR                  |                       $\lfloor x \rfloor$                       |    FP32/FP16/INT8    |
|                  LOG                   |                     $\log \left( x \right)$                     |    FP32/FP16/INT8    |
|                  NEG                   |                              $-x$                               |    FP32/FP16/INT8    |
|                  NOT                   |                     $not \left( x \right)$                      |         bool         |
|                 RECIP                  |                          $\frac{1}{x}$                          |    FP32/FP16/INT8    |
|                 ROUND                  |                      Round$\left(x\right)$                      |    FP32/FP16/INT8    |
|                  SIGN                  |               $\frac{1}{1+\exp{\left(-x\right)}}$               | FP32/FP16/INT8/INT32 |
|                  SQRT                  |                           $\sqrt{x}$                            |    FP32/FP16/INT8    |
|                  SIN                   |                     $\sin \left( x \right)$                     |    FP32/FP16/INT8    |
|                  COS                   |                     $\cos \left( x \right)$                     |    FP32/FP16/INT8    |
|                  TAN                   |                     $\tan \left( x \right)$                     |    FP32/FP16/INT8    |
|                  ASIN                  |                  $\sin^{-1} \left( x \right)$                   |    FP32/FP16/INT8    |
|                  ACOS                  |                  $\cos^{-1} \left( x \right)$                   |    FP32/FP16/INT8    |
|                  ATAN                  |                  $\tan^{-1} \left( x \right)$                   |    FP32/FP16/INT8    |
|                  SINH                  |                    $\sinh \left( x \right)$                     |    FP32/FP16/INT8    |
|                  COSH                  |                    $\cosh \left( x \right)$                     |    FP32/FP16/INT8    |
| $\color{#FF0000}{There\ is\ no\ TANH}$ |                use ActivationLayer to call tanh                 |    FP32/FP16/INT8    |
|                 ASINH                  |                  $\sinh^{-1} \left( x \right)$                  |    FP32/FP16/INT8    |
|                 ACOSH                  |                  $\cosh^{-1} \left( x \right)$                  |    FP32/FP16/INT8    |
|                 ATANH                  |                  $\tanh^{-1} \left( x \right)$                  |    FP32/FP16/INT8    |
|       ISINF (since TensorRT 8.6)       |                   1 if element == Inf else 0                    |    FP32/FP16/INT8    |