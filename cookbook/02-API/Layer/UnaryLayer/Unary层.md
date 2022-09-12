# Unary 层
+ 初始范例代码
+ op

---
### 初始范例代码
+ 见 SimpleUsage.py

+ 输入张量形状 (1,1,3,3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            -4. & -3. & -2. \\
            -1. &  0. &  1. \\
             2. &  3. &  4.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (1,1, 3, 3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            4. & 3. & 2. \\
            1. & 0. & 1. \\
            2. & 3. & 4.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### op
+ 见 Op.py，在构建 Unary 层后再修改其计算类型

+ 输出张量形状 (1,1,3,3)，结果与初始范例代码相同

+ 可用的一元函数
|        trt.UnaryOperation 名        |                             函数                             |
| :---------------------------------: | :----------------------------------------------------------: |
|                 NEG                 |                             $-x$                             |
|                 NOT                 |                    $not \left( x \right)$                    |
|                 ABS                 |                      $\left| x \right|$                      |
|                FLOOR                |                     $\lfloor x \rfloor$                      |
|                CEIL                 |                      $\lceil x \rceil$                       |
|                RECIP                |                        $\frac{1}{x}$                         |
|                SQRT                 |                         $ \sqrt{x} $                         |
|                 EXP                 |                   $\exp \left( x \right)$                    |
|                 LOG                 |            $ \log \left( x \right) $（以 e 为底）            |
|                 ERF                 | $ erf \left( x \right) = \int_{0}^{x} \exp\left(-t^{2}\right)dt$ |
|                 SIN                 |                   $\sin \left( x \right)$                    |
|                 COS                 |                   $\cos \left( x \right)$                    |
|                 TAN                 |                   $\tan \left( x \right)$                    |
|                ASIN                 |                 $\sin^{-1} \left( x \right)$                 |
|                ACOS                 |                 $\cos^{-1} \left( x \right)$                 |
|                ATAN                 |                 $\tan^{-1} \left( x \right)$                 |
|                SINH                 |                   $\sinh \left( x \right)$                   |
|                COSH                 |                   $\cosh \left( x \right)$                   |
| <font color=#FF0000>没有TANH</font> |                tanh 作为 activation 层的函数                 |
|                ASINH                |                $\sinh^{-1} \left( x \right)$                 |
|                ACOSH                |                $\cosh^{-1} \left( x \right)$                 |
|                ATANH                |                $\tanh^{-1} \left( x \right)$                 |

