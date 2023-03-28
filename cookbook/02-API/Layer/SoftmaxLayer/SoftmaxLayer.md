# SoftMax Layer
+ Simple example
+ axes

---
## Simple example
+ Refer to SimpleExample.py
+ Shape of input tensor 0: (1,3,3,3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 2. & 3. \\
            4. & 5. & 6. \\
            7. & 8. & 9.
        \end{matrix}\right]
        \left[\begin{matrix}
            10. & 11. & 12. \\
            13. & 14. & 15. \\
            16. & 17. & 18.
        \end{matrix}\right]
        \left[\begin{matrix}
            19. & 20. & 21. \\
            22. & 23. & 24. \\
            25. & 26. & 27.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (3, 3, 3)，默认在“非 batch 维的最高维”上计算 softmax，各通道相同 HW 位置上元素之和为 1
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            {\color{#007F00}{0.00000002}} & 0.00000002 & 0.00000002 \\
            0.00000002 & 0.00000002 & 0.00000002 \\
            0.00000002 & 0.00000002 & 0.00000002
        \end{matrix}\right]
        \left[\begin{matrix}
            {\color{#0000FF}{0.00012339}} & 0.00012339 & 0.00012339 \\
            0.00012339 & 0.00012339 & 0.00012339 \\
            0.00012339 & 0.00012339 & 0.00012339
        \end{matrix}\right]
        \left[\begin{matrix}
            {\color{#FF0000}{0.9998766}} & 0.9998766 & 0.9998766 \\
            0.9998766 & 0.9998766 & 0.9998766 \\
            0.9998766 & 0.9998766 & 0.9998766
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\frac{ \left[ e^{1},e^{10},e^{19} \right] }{ e^{1}+e^{10}+e^{19} }
=
\left[
    {\color{#007F00}{1.523 \times 10^{-8}}}, {\color{#0000FF}{1.234 \times 10^{-4}}}, {\color{#FF0000}{9.998 \times 10^{-1}}}
\right]
$$

---

## axes
+ 见 Op，在构建Softmax 层后再后再修改其求规约的维度

+ 指定 axes=1<<0（在最高维上计算 softmax）, shape of output tensor 0: (1,3,3,3)，只有一个元素参与，结果全都是 1
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 1. & 1. \\
            2. & 1. & 1. \\
            3. & 1. & 1.
        \end{matrix}\right]
        \left[\begin{matrix}
            1. & 1. & 1. \\
            2. & 1. & 1. \\
            3. & 1. & 1.
        \end{matrix}\right]
        \left[\begin{matrix}
            1. & 1. & 1. \\
            2. & 1. & 1. \\
            3. & 1. & 1.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 axes=1<<1（在次高维上计算 softmax）, shape of output tensor 0: (1,3,3,3)，结果与初始范例代码相同

+ 指定 axes=1<<2（在季高维上计算 softmax）, shape of output tensor 0: (1,3,3,3)，每个通道内同一列元素之和为 1
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            0.00235563 & 0.00235563 & 0.00235563 \\
            0.04731416 & 0.04731416 & 0.04731416 \\
            0.95033026 & 0.95033026 & 0.95033026
        \end{matrix}\right]
        \left[\begin{matrix}
            0.00235563 & 0.00235563 & 0.00235563 \\
            0.04731416 & 0.04731416 & 0.04731416 \\
            0.95033026 & 0.95033026 & 0.95033026
        \end{matrix}\right]
        \left[\begin{matrix}
            0.00235563 & 0.00235563 & 0.00235563 \\
            0.04731416 & 0.04731416 & 0.04731416 \\
            0.95033026 & 0.95033026 & 0.95033026]
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 axes=1<<3（在叔高维上计算 softmax）, shape of output tensor 0: (1,3,3,3)，每个通道内同一行元素之和为 1
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            0.09003057 & 0.24472848 & 0.66524094 \\
            0.09003057 & 0.24472848 & 0.66524094 \\
            0.09003057 & 0.24472848 & 0.66524094
        \end{matrix}\right]
        \left[\begin{matrix}
            0.09003057 & 0.24472848 & 0.66524094 \\
            0.09003057 & 0.24472848 & 0.66524094 \\
            0.09003057 & 0.24472848 & 0.66524094
        \end{matrix}\right]
        \left[\begin{matrix}
            0.09003057 & 0.24472848 & 0.66524094 \\
            0.09003057 & 0.24472848 & 0.66524094 \\
            0.09003057 & 0.24472848 & 0.66524094
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 不能同时指定两个及以上的 axes，如使用 axes=(1<<0)+(1<<1)，会收到报错
```
# TensorRT 8.2
[TRT] [E] 3: [layers.h::setAxes::600] Error Code 3: API Usage Error (Parameter check failed at: /_src/build/cuda-11.4/8.2/x86_64/release/optimizer/api/layers.h::setAxes::600, condition: isSingleBit(axes)
)
# TensorRT 8.4
[TRT] [E] 3: [layers.h::setAxes::599] Error Code 3: API Usage Error (Parameter check failed at: /_src/build/x86_64-gnu/release/optimizer/api/layers.h::setAxes::599, condition: isSingleBit(axes)
)
```

+ 没有指定 axes 时，其默认值为 1 << max{0, n-3}，n 为输入张量的维度数
