# RaggedSoftMax 层
+ 初始示例代码

---
### 初始示例代码
+ 见 SimpleUsage.py

+ 输入张量 0 形状 (3,4,5)
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

+ 输入张量 1 形状 (1,3,4,1)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0 \\
        2 \\
        4 \\
        6
    \end{matrix}\right]
    \left[\begin{matrix}
        0 \\
        2 \\
        4 \\
        6
    \end{matrix}\right]
    \left[\begin{matrix}
        0 \\
        2 \\
        4 \\
        6
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (3,4,5)，每个 batch 都在指定长度 (1,2,3,4) 上计算了 Soft Max，其余元素变成 0
+ 计算长度为 0 时输出值全为 0（每 batch 第一行），计算长度大于输入张量 1 的宽度时，存在访存越界（第 2 batch 最后一行红色数字），结果随机
+ 这里只是恰好 $0.1862933 = \frac{e^1}{5e^1+e^0}$
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0.    & 0.    & 0.    & 0.    & 0.    \\
        0.5   & 0.5   & 0.    & 0.    & 0.    \\
        0.25  & 0.25  & 0.25  & 0.25  & 0.    \\
        0.167 & 0.167 & 0.167 & 0.167 & 0.167 \\
    \end{matrix}\right] \\
    \left[\begin{matrix}
        0.    & 0.    & 0.    & 0.    & 0.    \\
        0.5   & 0.5   & 0.    & 0.    & 0.    \\
        0.25  & 0.25  & 0.25  & 0.25  & 0.    \\
        0.167 & 0.167 & 0.167 & 0.167 & 0.167 \\
    \end{matrix}\right] \\
    \left[\begin{matrix}
        0.    & 0.    & 0.    & 0.    & 0.    \\
        0.5   & 0.5   & 0.    & 0.    & 0.    \\
        0.25  & 0.25  & 0.25  & 0.25  & 0.    \\
        \textcolor[rgb]{1,0,0}{0.186} & \textcolor[rgb]{1,0,0}{0.186} & \textcolor[rgb]{1,0,0}{0.186} & \textcolor[rgb]{1,0,0}{0.186} & \textcolor[rgb]{1,0,0}{0.186}
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 该层两个输入张量只接受 3 维张量，否则报错：
```
[TRT] [E] 4: [raggedSoftMaxNode.cpp::computeOutputExtents::13] Error Code 4: Internal Error ((Unnamed Layer* 0) [Ragged SoftMax]: Input tensor must have exactly 3 dimensions)
[TRT] [E] 4: (Unnamed Layer* 0) [Ragged SoftMax]: input tensor must have 2 non batch dimensions
[TRT] [E] 4: [network.cpp::validate::2871] Error Code 4: Internal Error (Layer (Unnamed Layer* 0) [Ragged SoftMax] failed validation)
```

+ 两个输入的维度要一致，否则报错：
```
[TRT] [E] 3: [network.cpp::addRaggedSoftMax::1294] Error Code 3: API Usage Error (Parameter check failed at: optimizer/api/network.cpp::addRaggedSoftMax::1294, condition: input.getDimensions().nbDims == bounds.getDimensions().nbDims
```