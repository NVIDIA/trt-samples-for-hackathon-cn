# Fill 层
+ 初始范例代码
+ set_input + 构建期线性填充
+ set_input + 构建期均匀随机填充
+ 构建期指定形状 + 运行期指定范围的均匀随机填充
+ 运行期指定形状和范围的均匀随机填充
+ 设置随机种子（尚不可用）[TODO]

---
### 初始范例代码
+ 见 SimpleUsage.p 和 SimpleUsage2.py 

+ LINSPACE 模式输出张量形状 (1,3,4,5)，TensorRT8 建立网络失败，没有输出
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 0. & 0. & 0. & 0. \\
        2. & 0. & 0. & 0. & 0. \\
        3. & 0. & 0. & 0. & 0. \\
        4. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 0. & 0. & 0. & 0. \\
        2. & 0. & 0. & 0. & 0. \\
        3. & 0. & 0. & 0. & 0. \\
        4. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 0. & 0. & 0. & 0. \\
        2. & 0. & 0. & 0. & 0. \\
        3. & 0. & 0. & 0. & 0. \\
        4. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 包含错误，因为默认指定 LINSPACE 模式填充，但是没有指定起点张量（$\alpha$）和增量张量（$\beta$）

```shell
#TensorRT 7:
[TensorRT] ERROR: 2: [fillRunner.cpp::executeLinSpace::46] Error Code 2: Internal Error (Assertion dims.nbDims == 1 failed.Alpha and beta tensor should be set when output an ND tensor)
[TensorRT] INTERNAL ERROR: Assertion failed: dims.nbDims == 1 && "Alpha and beta tensor should be set when output an ND tensor"
#TensorRT 8:
[TRT] [E] 2: [fillRunner.cpp::executeLinSpace::46] Error Code 2: Internal Error (Assertion dims.nbDims == 1 failed. Alpha and beta tensor should be set when output an ND tensor)
```

+ RANDOM_UNIFORM 模式输出
$$
\left[\begin{matrix}
    \left[\begin{matrix}
    0.04162789 & 0.51853730 & 0.14025924 & 0.62825230 & 0.99067950 \\
    0.04355760 & 0.37655574 & 0.70721610 & 0.50495917 & 0.19941926 \\
    0.97111490 & 0.00185533 & 0.39105898 & 0.91859037 & 0.01803547 \\
    0.80530400 & 0.84825593 & 0.47534138 & 0.84691340 & 0.12121718
    \end{matrix}\right]
    \left[\begin{matrix}
    0.82836760 & 0.77331300 & 0.19260496 & 0.11180904 & 0.21402750 \\
    0.72633845 & 0.58760810 & 0.12656350 & 0.56200240 & 0.17877735 \\
    0.04002266 & 0.65710986 & 0.96338147 & 0.95881134 & 0.19407918 \\
    0.40907905 & 0.75002570 & 0.31181264 & 0.99177790 & 0.03078329
    \end{matrix}\right]
    \left[\begin{matrix}
    0.18007872 & 0.88346076 & 0.15291394 & 0.14343130 & 0.92315560 \\
    0.49598184 & 0.14866723 & 0.98038020 & 0.60230300 & 0.11726483 \\
    0.55478835 & 0.26550958 & 0.86637110 & 0.19065430 & 0.67076814 \\
    0.20564255 & 0.37991480 & 0.18683665 & 0.40659520 & 0.26049972
    \end{matrix}\right]
\end{matrix}\right]
$$

---

### set_input + 构建期线性填充

+ 见 Set_input+Linear.py，构建期线性填充

+ 输出张量形状 (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.    & 1001. & 1002. & 1003. & 1004. \\
            2.    & 1011. & 1012. & 1013. & 1014. \\
            3.    & 1021. & 1022. & 1023. & 1024. \\
            4.    & 1031. & 1032. & 1033. & 1034.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.    & 1101. & 1102. & 1103. & 1104. \\
            2.    & 1111. & 1112. & 1113. & 1114. \\
            3.    & 1121. & 1122. & 1123. & 1124. \\
            4.    & 1131. & 1132. & 1133. & 1134.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.    & 1201. & 1202. & 1203. & 1204. \\
            2.    & 1211. & 1212. & 1213. & 1214. \\
            3.    & 1221. & 1222. & 1223. & 1224. \\
            4.    & 1231. & 1232. & 1233. & 1234.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### set_input + 构建期均匀随机填充
+ 见 Set_input+Random.py，构建期均匀随机填充

+ 输出张量形状 (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            -9.167 &  0.371 & -7.195  & 2.565 &  9.814 \\
            -9.129 & -2.469 &  4.144  & 0.099 & -6.012 \\
             9.422 & -9.963 & -2.179  & 8.372 & -9.639 \\
             6.106 &  6.965 & -0.493  & 6.938 & -7.576
        \end{matrix}\right]
        \left[\begin{matrix}
             6.567 &  5.466 & -6.148 & -7.764 & -5.719 \\
             4.527 &  1.752 & -7.469 &  1.24  & -6.424 \\
            -9.2   &  3.142 &  9.268 &  9.176 & -6.118 \\
            -1.818 &  5.001 & -3.764 &  9.836 & -9.384
        \end{matrix}\right]
        \left[\begin{matrix}
            -6.398 &  7.669 & -6.942 & -7.131 &  8.463 \\
            -0.08  & -7.027 &  9.608 &  2.046 & -7.655 \\
             1.096 & -4.69  &  7.327 & -6.187 &  3.415 \\
            -5.887 & -2.402 & -6.263 & -1.868 & -4.79
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意：
  - 随机数种子固定，按相同形状和数值范围生成的随机数是相同的
  - 建立 engine 后其数值为常数，多次运行数值均相同

---
### 构建期指定形状 + 运行期指定范围的均匀随机填充
+ 见 BuildtimeShape+RuntimeRange+Random.py

+ 输出张量形状 (1,3,4,5)，结果与“set_input 与均匀随机填充”范例相同

---
### 运行期指定形状和范围的均匀随机填充
+ 见 RuntimeShapeRange+Random.py

+ 输出张量形状 (1,3,4,5)，结果与“set_input 与均匀随机填充”范例相同

---
### 设置随机种子（尚不可用）
