# Element Wise 层
+ 初始范例代码
+ op
+ 广播操作

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
\\
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2.
        \end{matrix}\right]
        \left[\begin{matrix}
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2.
        \end{matrix}\right]
        \left[\begin{matrix}
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (1,3,4,5)，两个输入张量做逐元素加法
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3.
        \end{matrix}\right]
        \left[\begin{matrix}
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3.
        \end{matrix}\right]
        \left[\begin{matrix}
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### op
+ 见 Op.py，在构建 Elementwise 层后再修改其计算类型

+ 输出张量形状 (1,3,4,5)，每个元素变成 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0.
        \end{matrix}\right]
        \left[\begin{matrix}
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0.
        \end{matrix}\right]
        \left[\begin{matrix}
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 可用的计算类型
| trt.ElementWiseOperation 名 |  $f\left(a,b\right)$   |          备注           |
| :-------------------------: | :--------------------: | :---------------------: |
|             SUM             |         a + b          |                         |
|            PROD             |         a * b          |                         |
|             MAX             | $\max\left(a,b\right)$ |                         |
|             MIN             | $\min\left(a,b\right)$ |                         |
|             SUB             |         a - b          |                         |
|             DIV             |         a / b          |                         |
|             POW             |   a \*\* b ($a^{b}$)   | 输入 float32/float16 型 |
|          FLOOR_DIV          |         a // b         |                         |
|             AND             |        a and b         |  输入输出都是 Bool 型   |
|             OR              |         a or b         |  输入输出都是 Bool 型   |
|             XOR             |    a ^ b (a xor b)     |  输入输出都是 Bool 型   |
|            EQUAL            |         a == b         |     输出是 Bool 型      |
|           GREATER           |         a > b          |     输出是 Bool 型      |
|            LESS             |         a < b          |     输出是 Bool 型      |

+ 需要 BOOL 型输入的算子在输入其他数据类型时报错：
```
[TensorRT] ERROR: 4: [layers.cpp::validate::2304] Error Code 4: Internal Error ((Unnamed Layer* 0) [ElementWise]: operation AND requires boolean inputs.)
```

+ 添加 float/int 与 bool 类型转换的 Idenetity 层时注意设置 config.set_memory_pool_limit，否则报错：
```
[TensorRT] ERROR: 1: [codeGenerator.cpp::createMyelinGraph::314] Error Code 1: Myelin (myelinTargetSetPropertyMemorySize called with invalid memory size (0).)
```

+ POW 的两个输入必须是 activation type，否则报错：
```
[TensorRT] ERROR: 4: [layers.cpp::validate::2322] Error Code 4: Internal Error ((Unnamed Layer* 0) [ElementWise]: operation POW requires inputs with activation type.)
# 或者
[TensorRT] ERROR: 4: [layers.cpp::validate::2291] Error Code 4: Internal Error ((Unnamed Layer* 0) [ElementWise]: operation POW has incompatible input types Float and Int32)
```

### 广播操作
+ 见 Broadcast.py，在逐元素计算时使用广播特性

+ 输入张量形状 (1,3,1,5) 和 (1,1,4,1)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
\\
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            2. \\ 2. \\ 2. \\ 2.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (1,3,4,5)，将两加数广播后进行计算，结果与初始范例代码相同