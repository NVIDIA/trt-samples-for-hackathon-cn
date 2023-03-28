# Parametric ReLU 层
+ Simple example

---
## Simple example
+ Refer to SimpleExample.py
+ Shape of input tensor 0: (1,3,3,3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            -4. & -3. & -2. \\
            -1. &  0. &  1. \\
             2. &  3. &  4.
        \end{matrix}\right]
        \left[\begin{matrix}
            -4. & -3. & -2. \\
            -1. &  0. &  1. \\
             2. &  3. &  4.
        \end{matrix}\right]
        \left[\begin{matrix}
            -4. & -3. & -2. \\
            -1. &  0. &  1. \\
             2. &  3. &  4.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (1,3,3,3)，就是 leaky ReLU，负数部分的斜率被改变
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            -2.  & -1.5. & -1. \\
            -0.5 &  0.   &  1. \\
             2.  &  3.   &  4.
        \end{matrix}\right]
        \left[\begin{matrix}
            -4. & -3. & -2. \\
            -1. &  0. &  1. \\
             2. &  3. &  4.
        \end{matrix}\right]
        \left[\begin{matrix}
            -8. & -6. & -4. \\
            -2. &  0. &  1. \\
             2. &  3. &  4.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 很多有 ReLU 参与的 Layer fusion 中没有对应的 Parametric ReLU 版本，使用 Parametric ReLU 性能可能比 ReLU 更差