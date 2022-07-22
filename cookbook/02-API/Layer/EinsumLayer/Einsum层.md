# Einsum 层
+ **Eisum 层从 TensorRT 8.2 开始支持，但是功能尚不完善**
+ 初始示例代码（用 Einsum 层做双张量单缩并）
+ equatiuon
+ 用 Einsum 层做转置
+ 用 Einsum 层做求和规约
+ 用 Einsum 层做点积
+ 用 Einsum 层做矩阵乘法
+ 用 Einsum 层做多张量缩并（尚不可用）[TODO]
+ 用 Einsum 层取对角元素（尚不可用）[TODO]
+ 省略号（...）用法（尚不可用）[TODO]

---
### 初始示例代码（用 Einsum 层做双张量单缩并）
+ 见 SimpleUsage.py

+ 输入张量形状 (1,3,4) 和 (2,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. &  3. \\
         4. &  5. &  6. &  7. \\
         8. &  9. & 10. & 11.
    \end{matrix}\right]
\end{matrix}\right]
,
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
         5. &  6. &  7. &  8. &  9. \\
        10. & 11. & 12. & 13. & 14. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        15. & 16. & 17. & 18. & 19. \\
        20. & 21. & 22. & 23. & 24. \\
        25. & 26. & 27. & 28. & 29.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (1,4,2,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
         100. & 112. & 124. & 136. & 148. \\
         280. & 292. & 304. & 316. & 328.
        \end{matrix}\right] \\
        \left[\begin{matrix}
         115. & 130. & 145. & 160. & 175. \\
         340. & 355. & 370. & 385. & 400.
        \end{matrix}\right] \\
        \left[\begin{matrix}
         130. & 148. & 166. & 184. & 202. \\
         400. & 418. & 436. & 454. & 472.
        \end{matrix}\right] \\
        \left[\begin{matrix}
         145. & 166. & 187. & 208. & 229. \\
         460. & 481. & 502. & 523. & 544.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：$A_{1\times3\times4}, B_{2\times3\times5}$ 关于长度为 3 的那一维度作缩并，输出为 $E_{1\times4\times2\times5}$
$$
\begin{aligned}
C_{1\times4\times3}         &= A^{\text{T}(0,2,1)} \\
D_{1\times2\times4\times5}  &= CB \\
E_{1\times4\times2\times5}  &= D^{\text{T}(0,2,1,3)}
\end{aligned}
$$

---
### equation
+ 见 Equation.py，在构建 Einsum 层后再修改其计算方程

+ 输出张量形状 (1,4,2,5)，结果与初始示例代码相同

---
### 用 Einsum 层做转置
+ 见 Transpose.py，做张量转置

+ 输入张量形状 (1,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
      0. &  1. &  2. &  3. \\
      4. &  5. &  6. &  7. \\
      8. &  9. & 10. & 11. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (3,4,1)，结果等价于 inputH0.transpose(1,2,0)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
      0. \\  1. \\  2. \\  3. \\
    \end{matrix}\right]
    \left[\begin{matrix}
      4. \\  5. \\  6. \\  7. \\
    \end{matrix}\right]
    \left[\begin{matrix}
      8. \\  9. \\ 10. \\ 11. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### 用 Einsum 层做求和规约
+ 见 Reduce.py，做单张量缩并

+ 输入张量形状 (1,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
      0. &  1. &  2. &  3. \\
      4. &  5. &  6. &  7. \\
      8. &  9. & 10. & 11. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定求和表达式 "ijk->ij"，输出张量形状 (1,3)，“->” 右侧最后一维 k 消失，即按该维求规约，结果等价于 np.sum(ipnutH0,axis=2)
$$
\left[\begin{matrix}
     6. & 22. & 38.
\end{matrix}\right]
$$

+ 指定求和表达式 "ijk->ik"，输出张量形状 (1,4)，结果等价于 np.sum(ipnutH0,axis=1)
$$
\left[\begin{matrix}
    12. & 15. & 18. & 21.
\end{matrix}\right]
$$

---
### 用 Einsum 层做点积
+ 见 Dot.py，做双张量点积

+ 输入张量形状 (1,1,4) 和 (1,1,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. &  1. &  2. &  3.
    \end{matrix}\right]
\end{matrix}\right]
,
\left[\begin{matrix}
    \left[\begin{matrix}
        1. &  1. &  1. &  1.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 ()
$$
6.0
$$

+ 计算过程，“->” 左侧相同下标（k）对应的维参与缩并运算（同维度内乘加），右侧消失的下标（i、j、p、q）都参与求和运算（跨维度求和）
$$
0.0 * 1.0 + 0.1 * 1.0 + 2.0 * 1.0 + 0.3 * 1.0 = 6.0
$$

```python
# 修改输入数据形状
nB0,nH0,nW0 = 1,2,4
nB1,nH1,nW1 = 1,3,4
```

+ 输入张量形状 (1,2,4) 和 (1,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. &  1. &  2. &  3. \\
        4. &  5. &  6. &  7.
    \end{matrix}\right]
\end{matrix}\right]
,
\left[\begin{matrix}
    \left[\begin{matrix}
        1. &  1. &  1. &  1. \\
        1. &  1. &  1. &  1. \\
        1. &  1. &  1. &  1.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 ()
$$
84.0
$$

+ 计算过程，每个张量最后一维参与内积计算，一共有 2 * 3 组
$$
6 + 6 + 6 + 22 + 22 + 22 = 84
$$

```python
# 同时修改输入数据形状和计算表达式
nB0,nH0,nW0 = 1,2,4
nB1,nH1,nW1 = 1,3,4

einsumLayer = network.add_einsum([inputT0, inputT1], "ijk,pqk->j")
```

+ 输出张量形状 (2,)
$$
\left[\begin{matrix}18. & 66.\end{matrix}\right]
$$

+ 计算过程，保留了 j = 2 这一下标不做加和，其他同上一个示例
$$
6 + 6 + 6 = 18，22 + 22 + 22 = 66
$$

---
### 用 Einsum 层做矩阵乘法
+ 见 MatrixMultiplication.py，做矩阵乘法

+ 输入张量形状 (2,2,3) 和 (2,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. \\
         3. &  4. &  5.
    \end{matrix}\right]
    \left[\begin{matrix}
         6. &  7. &  8. \\
         9. & 10. & 11.
    \end{matrix}\right]
\end{matrix}\right]
,
\left[\begin{matrix}
    \left[\begin{matrix}
         1. &  1. &  1. &  1. \\
         1. &  1. &  1. &  1. \\
         1. &  1. &  1. &  1.
    \end{matrix}\right]
    \left[\begin{matrix}
         1. &  1. &  1. &  1. \\
         1. &  1. &  1. &  1. \\
         1. &  1. &  1. &  1.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (2,2,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         3. &  3. &  3. &  3. \\
        12. & 12. & 12. & 12.
    \end{matrix}\right]
    \left[\begin{matrix}
        21. & 21. & 21. & 21. \\
        30. & 30. & 30. & 30.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程，在最高 i 维（batch 维）保留，每个 batch 内做矩阵乘法
$$
\left[\begin{matrix}
     0. &  1. &  2. \\
     3. &  4. &  5.
\end{matrix}\right]
\left[\begin{matrix}
     1. &  1. &  1. &  1. \\
     1. &  1. &  1. &  1. \\
     1. &  1. &  1. &  1.
\end{matrix}\right]
=
\left[\begin{matrix}
     3. &  3. &  3. &  3. \\
    12. & 12. & 12. & 12.
\end{matrix}\right]
$$

---
### 用 Einsum 层做多张量缩并（尚不可用）
+ 见 TripleTensor.py，多三个张量做缩并

+ TensorRT 8.4 中收到报错：
```
[TRT] [E] 3: [layers.cpp::EinsumLayer::5525] Error Code 3: API Usage Error (Parameter check failed at: optimizer/api/layers.cpp::EinsumLayer::5525, condition: nbInputs > 0 && nbInputs <= MAX_EINSUM_NB_INPUTS
```

---
### 用 Einsum 层取对角元素（尚不可用）[TODO]
+ 见 Diagonal.py，取张量对角元素

+ TensorRT 8.4 中收到报错：
```
[TRT] [E] 3: [layers.cpp::validateEquation::5611] Error Code 3: Internal Error ((Unnamed Layer* 0) [Einsum]: Diagonal operations are not permitted in Einsum equation)
```

---
### 省略号（...）用法（尚不可用）
+ 见 Ellipsis.py，使用省略号语法

+ TensorRT 8.4 中收到报错：
```
[TRT] [E] 3: [layers.cpp::validateEquation::5589] Error Code 3: Internal Error ((Unnamed Layer* 0) [Einsum]: ellipsis is not permitted in Einsum equation)
```
