# Reduce 层
+ 初始范例代码
+ op
+ axes
+ keep_dims

---
### 初始范例代码
+ 见 SimpleUsage.py

+ 输入张量形状 (1,3,4,5)
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

+ 输出张量形状 (1,4,5)，在次高维上进行了求和
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
### op
+ 见 Op.py，在构建 Reduce 层后再后再修改其求规约的算符

+ 输出张量形状 (1,4,5)，结果与初始范例代码相同

+ 可用的规约计算方法
| trt.ReduceOperation |   函数   |
| :-----------------: | :------: |
|        PROD         |   求积   |
|         AVG         | 求平均值 |
|         MAX         | 取最大值 |
|         MIN         | 取最小值 |
|         SUM         |   求和   |

---
### axes
+ 见 Axes.py，在构建 Reduce 层后再后再修改其求规约的维度

+ 指定 axes=1<<0，输出张量形状 (3,4,5)，在最高维上进行规约，相当于什么也没做
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

+ 指定 axes=1<<1，输出张量形状 (1,4,5)，在次高维上进行规约，结果与初始范例代码相同

+ 指定 axes=1<<2，输出张量形状 (1,3,5)，在季高维上进行规约
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        4. & 4. & 4. & 4. & 4. \\
        4. & 4. & 4. & 4. & 4. \\
        4. & 4. & 4. & 4. & 4.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 axes=(1<<2)+(1<<3)，输出张量形状 (1,3)，同时在第二和第三维度上进行规约，注意 << 优先级低于 +，要加括号
$$
\left[\begin{matrix}
    20. & 20. & 20.
\end{matrix}\right]
$$

---
### keep_dims
+ 见 Keep_dims.py，是否要保留输出张量中被规约维度的 1 维

+ 指定 keep_dims=True，输出张量形状 (1,1,4,5)，保留了求规约的维度
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

+ 指定 keep_dims=False，输出张量形状 (1,4,5)，结果与初始范例代码相同