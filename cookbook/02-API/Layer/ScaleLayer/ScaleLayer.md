# Scale Layer
+ Simple example
+ mode & scale & shift & power
+ CHANNEL 和 ELEMENTWISE 级的 scale
+ add_scale_nd 及其参数 channel_axis

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

+ Shape of output tensor 0: (1,3,3,3)，所有元素都做了变换 $ y = \left( x \cdot scale + shift\right) ^{power} $
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            -6.5 & -6.  & -5.5 \\
            -5.  & -4.5 & -4.  \\
            -3.5 & -3.  & -2.5
        \end{matrix}\right]
        \left[\begin{matrix}
            -2.  & -1.5 & -1.  \\
            -0.5 &  0.  &  0.5 \\
             1.  &  1.5 &  2.
        \end{matrix}\right]
        \left[\begin{matrix}
            2.5  & 3.  & 3.5 \\
            4.   & 4.5 & 5.  \\
            5.5  & 6.  & 6.5
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ TensorRT6 及之前版本需要对参数 scale，shift，power 的 np 数组作拷贝，防止其被后续同名变量覆盖，因为 TensorRT 中这些参数的定义和使用是异步的。TensorRT7 及之后该问题被修正，不再需要额外的拷贝工作
```python
# TensorRT 6
bag = []
scale = np.ascontiguousarray(np.array([0.5], dtype=np.float32))
shift = np.ascontiguousarray(np.array([-7.0], dtype=np.float32))
power = np.ascontiguousarray(np.array([1.0], dtype=np.float32))
bag += [scale, shift, power]
scaleLaer = network.add_scale(...)
```

---

## mode & scale & shift & power
+ Refer to Mode+Scale+Shift+Power.py，构建 Scale 层后再修改其模式和参数

+ Shape of output tensor 0: (3,3,3)，与初始范例代码相同

+ add_scale 中的 scale，shift，power 参数可传入 None，此时将使用他们的默认值，分别为 1.0，0.0，1.0

+ 可用的模式
| trt.ScaleMode 名 |            说明             |
| :--------------: | :-------------------------: |
|   ELEMENTWISE    |    每个元素使用一套参数     |
|     UNIFORM      |    所有元素使用一套参数     |
|     CHANNEL      | 同个 C 维的元素使用一套参数 |

---

## CHANNEL 和 ELEMENTWISE 级的 scale
+ Refer to ModeChannel.py 和 ModeElement.py
+ Channel 模式, shape of output tensor 0: (1,3,3,3)，每个通道依不同参数进行 scale
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            -2.  & -1.5 & -1.  \\
            -0.5 &  0.  &  0.5 \\
             1.  &  1.5 &  2.
        \end{matrix}\right]
        \left[\begin{matrix}
            -2.  & -1.5 & -1.  \\
            -0.5 &  0.  &  0.5 \\
             1.  &  1.5 &  2.
        \end{matrix}\right]
        \left[\begin{matrix}
            -2.  & -1.5 & -1.  \\
            -0.5 &  0.  &  0.5 \\
             1.  &  1.5 &  2.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Element 模式, shape of output tensor 0: (1,3,3,3)，每个元素依不同参数进行 scale，结果与初始范例代码相同

---

## 使用 add_scale_nd 和 channel_axis
+ Refer to Add_scale_nd+Channel_axis.py
+ Shape of input tensor 0: (2,2,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             1. &  0. &  0. &  0.\\
             2. &  0. &  0. &  0.\\
             3. &  0. &  0. &  0.
        \end{matrix}\right]
        \left[\begin{matrix}
             1. &  0. &  0. &  0.\\
             2. &  0. &  0. &  0.\\
             3. &  0. &  0. &  0.
        \end{matrix}\right]
    \end{matrix}\right]
    \\
    \left[\begin{matrix}
        \left[\begin{matrix}
             1. &  0. &  0. &  0.\\
             2. &  0. &  0. &  0.\\
             3. &  0. &  0. &  0.
        \end{matrix}\right]
        \left[\begin{matrix}
             1. &  0. &  0. &  0.\\
             2. &  0. &  0. &  0.\\
             3. &  0. &  0. &  0.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 channel_axis=0（在 N 维上进行 scale）, shape of output tensor 0: (2, 2, 3, 4)，每个 batch 内 2 个通道共用一套参数
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             2. &  2. &  2. &  2.\\
             2. &  2. &  2. &  2.\\
             2. &  2. &  2. &  2.
        \end{matrix}\right]
        \left[\begin{matrix}
             2. &  2. &  2. &  2.\\
             2. &  2. &  2. &  2.\\
             2. &  2. &  2. &  2.
        \end{matrix}\right]
    \end{matrix}\right]
    \\
    \left[\begin{matrix}
        \left[\begin{matrix}
             3. &  3. &  3. &  3.\\
             3. &  3. &  3. &  3.\\
             3. &  3. &  3. &  3.
        \end{matrix}\right]
        \left[\begin{matrix}
             3. &  3. &  3. &  3.\\
             3. &  3. &  3. &  3.\\
             3. &  3. &  3. &  3.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 channel_axis=1（在 C 维上进行 scale）, shape of output tensor 0: (2, 2, 3, 4)，各 batch 内同一通道共用一套参数
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             2. &  2. &  2. &  2.\\
             2. &  2. &  2. &  2.\\
             2. &  2. &  2. &  2.
        \end{matrix}\right]
        \left[\begin{matrix}
             3. &  3. &  3. &  3.\\
             3. &  3. &  3. &  3.\\
             3. &  3. &  3. &  3.
        \end{matrix}\right]
    \end{matrix}\right]
    \\
    \left[\begin{matrix}
        \left[\begin{matrix}
             2. &  2. &  2. &  2.\\
             2. &  2. &  2. &  2.\\
             2. &  2. &  2. &  2.
        \end{matrix}\right]
        \left[\begin{matrix}
             3. &  3. &  3. &  3.\\
             3. &  3. &  3. &  3.\\
             3. &  3. &  3. &  3.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$
