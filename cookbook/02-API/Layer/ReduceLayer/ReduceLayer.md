# Reduce 层
+ Simple example
+ op
+ axes
+ keep_dims

---
## Simple example
+ Refer to SimpleExample.py
+ Shape of input tensor 0: (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (1,4,5)，在次高维上进行了求和
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3.
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## op
+ Refer to Op.py，在构建 Reduce 层后再后再修改其求规约的算符

+ Shape of output tensor 0: (1,4,5)，结果与初始范例代码相同

+ 可用的规约计算方法
| trt.ReduceOperation |   函数   |
| :-----------------: | :------: |
|         SUM         |   求和   |
|        PROD         |   求积   |
|         AVG         | 求平均值 |
|         MAX         | 取最大值 |
|         MIN         | 取最小值 |

+ 对空张量进行规约计算的结果为特定的值
| trt.ReduceOperation | float32 / float16 |  int32  | int8  |
| :-----------------: | :---------------: | :-----: | :---: |
|        kSUM         |         0         |    0    |   0   |
|        kPROD        |         1         |    1    |   1   |
|        kMAX         |     $\infty$      | INT_MAX | -128  |
|        kMIN         |     $-\infty$     | INT_MIN |  127  |
|        kAVG         |        NaN        |    0    | -128  |


---

## axes
+ Refer to Axes.py，在构建 Reduce 层后再后再修改其求规约的维度

+ 指定 axes=1<<0, shape of output tensor 0: (3,4,5)，在最高维上进行规约，相当于什么也没做
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 axes=1<<1, shape of output tensor 0: (1,4,5)，在次高维上进行规约，结果与初始范例代码相同

+ 指定 axes=1<<2, shape of output tensor 0: (1,3,5)，在季高维上进行规约
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        4. & 4. & 4. & 4. & 4. \\
        4. & 4. & 4. & 4. & 4. \\
        4. & 4. & 4. & 4. & 4.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 axes=(1<<2)+(1<<3), shape of output tensor 0: (1,3)，同时在第二和第三维度上进行规约，注意 << 优先级低于 +，要加括号
$$
\left[\begin{matrix}
    20. & 20. & 20.
\end{matrix}\right]
$$

---

## keep_dims
+ Refer to Keep_dims.py，是否要保留输出张量中被规约维度的 1 维

+ 指定 keep_dims=True, shape of output tensor 0: (1,1,4,5)，保留了求规约的维度
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 keep_dims=False, shape of output tensor 0: (1,4,5)，结果与初始范例代码相同
