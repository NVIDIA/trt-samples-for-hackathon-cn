# LRN 层
+ 初始范例代码
+ window_size & alpha & beta & k

---
### 初始范例代码
+ 见 SimpleUsage.py

+ 输入张量形状 (1,3,3,3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 1. & 1. \\
            1. & 1. & 1. \\
            1. & 1. & 1.
        \end{matrix}\right]
        \left[\begin{matrix}
            2. & 2. & 2. \\
            2. & 2. & 2. \\
            2. & 2. & 2.
        \end{matrix}\right]
        \left[\begin{matrix}
            5. & 5. & 5. \\
            5. & 5. & 5. \\
            5. & 5. & 5.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (1,3,3,3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \textcolor[rgb]{0,0.5,0}{0.59996396} & 0.59996396 & 0.59996396 \\
            0.59996396 & 0.59996396 & 0.59996396 \\
            0.59996396 & 0.59996396 & 0.59996396
        \end{matrix}\right]
        \\
        \left[\begin{matrix}
            \textcolor[rgb]{0,0,1}{0.19999799} & 0.19999799 & 0.19999799 \\
            0.19999799 & 0.19999799 & 0.19999799 \\
            0.19999799 & 0.19999799 & 0.19999799
        \end{matrix}\right]
        \\
        \left[\begin{matrix}
            \textcolor[rgb]{1,0,0}{0.51723605} & 0.51723605 & 0.51723605 \\
            0.51723605 & 0.51723605 & 0.51723605 \\
            0.51723605 & 0.51723605 & 0.51723605
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：$n=3,\alpha=\beta=1.,k=0.0001$，求和元素个数等于 $\lfloor \frac{n}{2} \rfloor$，超出输入张量边界的部分按 0 计算
$$
\frac{1^{2}}{ \left( k + \frac{\alpha}{3} \left( 0^{2} + 1^{2} + 2^{2} \right) \right)^{\beta} }
= \textcolor[rgb]{0,0.5,0}{0.59996396},
\\
\frac{2^{2}}{ \left( k + \frac{\alpha}{3} \left( 1^{2} + 2^{2} + 5^{2} \right) \right)^{\beta} }
= \textcolor[rgb]{0,0,1}{0.19999799},
\\
\frac{5^{2}}{ \left( k + \frac{\alpha}{3} \left( 2^{2} + 5^{2} + 0^{2}\right) \right)^{\beta} }
= \textcolor[rgb]{1,0,0}{0.51723605}
$$

---
### window_size & alpha & beta & k
+ 见 Window_size+Alpha+Bbeta+K.py，在构建 LRN 层后再修改其窗口大小和参数值

+ 输出张量形状 (1,3,3,3)，结果与初始范例代码相同