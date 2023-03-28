# Scatter Layer
+ mode（since TensorRT 8.2）
    - Scatter ELEMENT 模式
    - Scatter ND 模式

---

## mode（since TensorRT 8.2）

### ELEMENT 模式
+ Refer to ModeElement.py
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             0. &  1. &  2. &  3. &  4. \\
             5. &  6. &  7. &  8. &  9. \\
            10. & 11. & 12. & 13. & 14. \\
            15. & 16. & 17. & 18. & 19.
        \end{matrix}\right]
        \left[\begin{matrix}
            20. & 21. & 22. & 23. & 24. \\
            25. & 26. & 27. & 28. & 29. \\
            30. & 31. & 32. & 33. & 34. \\
            35. & 36. & 37. & 38. & 39.
        \end{matrix}\right]
        \left[\begin{matrix}
            40. & 41. & 42. & 43. & 44. \\
            45. & 46. & 47. & 48. & 49. \\
            50. & 51. & 52. & 53. & 54. \\
            55. & 56. & 57. & 58. & 59.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of input tensor $1: (1, 3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            0 & 1 & 2 & 3 & 0 \\
            1 & 2 & 3 & 0 & 1 \\
            2 & 3 & 0 & 1 & 2 \\
            3 & 0 & 1 & 2 & 3
        \end{matrix}\right]
        \left[\begin{matrix}
            0 & 1 & 2 & 3 & 0 \\
            1 & 2 & 3 & 0 & 1 \\
            2 & 3 & 0 & 1 & 2 \\
            3 & 0 & 1 & 2 & 3
        \end{matrix}\right]
        \left[\begin{matrix}
            0 & 1 & 2 & 3 & 0 \\
            1 & 2 & 3 & 0 & 1 \\
            2 & 3 & 0 & 1 & 2 \\
            3 & 0 & 1 & 2 & 3
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output  0  tensor: (1, 3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             -0. & -16. & -12. &  -8. &  -4. \\
             -5. &  -1. & -17. & -13. &  -9. \\
            -10. &  -6. &  -2. & -18. & -14. \\
            -15. & -11. &  -7. &  -3. & -19.
        \end{matrix}\right]
        \left[\begin{matrix}
            -20. & -36. & -32. & -28. & -24. \\
            -25. & -21. & -37. & -33. & -29. \\
            -30. & -26. & -22. & -38. & -34. \\
            -35. & -31. & -27. & -23. & -39.
        \end{matrix}\right]
        \left[\begin{matrix}
            -40. & -56. & -52. & -48. & -44. \\
            -45. & -41. & -57. & -53. & -49. \\
            -50. & -46. & -42. & -58. & -54. \\
            -55. & -51. & -47. & -43. & -59.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 含义：参考 [Onnx ScatterElements 算子](https://github.com/onnx/onnx/blob/main/docs/Operators.md#scatterelements) 和 [TensorRT C++ API 说明](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_scatter_layer.html)
+ 数据张量 data、索引张量 index、更新张量 update、输出张量 output 形状相同（$dim=r$），均为 $[d_{0},d_{1},...,d_{r-1}]$，指定 $axis=p$（$0 \le p < r$），则
+ 命循环变量 $i_{j}$ 满足 $ 0 \le j < r, 0 \le i_{j} < d_{j}$，则计算过程用 numpy 语法可以写作：$output[i_{0},i_{1},...,i_{p-1},index[i_{0},i_{1},...,i_{p-1},i_{p},i_{p+1},...,i_{r-1}],i_{p+1},...,i_{r-1}] = data[i_{0},i_{1},...,i_{p-1},i_{p},i_{p+1},...,i_{r-1}]$
+ 对于上面的范例代码，就是：
$$
index[0,0,0,0] = {\color{#FF0000}{0}} \Rightarrow output[0,0,{\color{#FF0000}{0}},0] = update[0,0,0,0] = -0. \\
index[0,0,0,1] = {\color{#FF0000}{1}} \Rightarrow output[0,0,{\color{#FF0000}{1}},1] = update[0,0,0,1] = -1. \\
index[0,0,0,2] = {\color{#FF0000}{2}} \Rightarrow output[0,0,{\color{#FF0000}{2}},2] = update[0,0,0,2] = -2. \\
\cdots \\
index[0,0,1,0] = {\color{#007F00}{1}} \Rightarrow output[0,0,{\color{#007F00}{1}},0] = update[0,0,1,0] = -5. \\
index[0,0,1,1] = {\color{#007F00}{2}} \Rightarrow output[0,0,{\color{#007F00}{2}},1] = update[0,0,2,1] = -6. \\
index[0,0,1,2] = {\color{#007F00}{3}} \Rightarrow output[0,0,{\color{#007F00}{3}},2] = update[0,0,3,2] = -7. \\
\cdots \\
index[0,1,0,0] = {\color{#0000FF}{0}} \Rightarrow output[0,1,{\color{#0000FF}{0}},0] = update[0,1,0,0] = -20. \\
index[0,1,0,1] = {\color{#0000FF}{1}} \Rightarrow output[0,1,{\color{#0000FF}{1}},1] = update[0,1,0,1] = -21. \\
index[0,1,0,2] = {\color{#0000FF}{2}} \Rightarrow output[0,1,{\color{#0000FF}{2}},2] = update[0,1,0,2] = -22. \\
\cdots \\
$$

+ 计算公式恰好为 GatherElement 算子公式等号左右两项的索引进行交换
+ output 元素的更新没有次序保证。如果两次更新指向 output 同一位置，且两次更新的值不同，则不能保证 output 该位置上的值选哪一次更新的结果。例如，将范例代码中 data1 改为 ```data1 = np.zeros([nB, nC, nH, nW], dtype=np.int32)```，那么 output[:,:,0,:] 值会是来自 update 的负的随机整数
        
---

### ND 模式
+ Refer to ModeND.py 和 ModeND2.py
+ Shape of input tensor $1: (2, 3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             0. &  1. &  2. &  3. &  4. \\
             5. &  6. &  7. &  8. &  9. \\
            10. & 11. & 12. & 13. & 14. \\
            15. & 16. & 17. & 18. & 19.
        \end{matrix}\right]
        \left[\begin{matrix}
            20. & 21. & 22. & 23. & 24. \\
            25. & 26. & 27. & 28. & 29. \\
            30. & 31. & 32. & 33. & 34. \\
            35. & 36. & 37. & 38. & 39.
        \end{matrix}\right]
        \left[\begin{matrix}
            40. & 41. & 42. & 43. & 44. \\
            45. & 46. & 47. & 48. & 49. \\
            50. & 51. & 52. & 53. & 54. \\
            55. & 56. & 57. & 58. & 59.
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            60. & 61. & 62. & 63. & 64. \\
            65. & 66. & 67. & 68. & 69. \\
            70. & 71. & 72. & 73. & 74. \\
            75. & 76. & 77. & 78. & 79.
        \end{matrix}\right]
        \left[\begin{matrix}
            80. & 81. & 82. & 83. & 84. \\
            85. & 86. & 87. & 88. & 89. \\
            90. & 91. & 92. & 93. & 94. \\
            95. & 96. & 97. & 98. & 99.
        \end{matrix}\right]
        \left[\begin{matrix}
            100. & 101. & 102. & 103. & 104. \\
            105. & 106. & 107. & 108. & 109. \\
            110. & 111. & 112. & 113. & 114. \\
            115. & 116. & 117. & 118. & 119.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of input tensor $1: (2, 3, 4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0 & 2 & 1 & 1 \\
        1 & 0 & 3 & 2 \\
        0 & 1 & 2 & 3
    \end{matrix}\right]
    \left[\begin{matrix}
        1 & 2 & 1 & 1 \\
        0 & 0 & 3 & 2 \\
        1 & 1 & 2 & 3
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of input tensor $1: (2, 3)
$$
\left[\begin{matrix}
    -0. & -1. & -2. \\
    -3. & -4. & -5.
\end{matrix}\right]
$$

+ Shape of output  0  tensor: (2, 3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             0. &  1. &  2. &  3. &  4. \\
             5. &  6. &  7. &  8. &  9. \\
            10. & 11. & 12. & 13. & 14. \\
            15. & 16. & -4. & 18. & 19.
        \end{matrix}\right]
        \left[\begin{matrix}
            20. & 21. & 22. & 23. & 24. \\
            25. & 26. & 27. & 28. & 29. \\
            30. & 31. & 32. & -2. & 34. \\
            35. & 36. & 37. & 38. & 39.
        \end{matrix}\right]
        \left[\begin{matrix}
            40. & 41. & 42. & 43. & 44. \\
            45. & -0. & 47. & 48. & 49. \\
            50. & 51. & 52. & 53. & 54. \\
            55. & 56. & 57. & 58. & 59.
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            60. & 61. & 62. & 63. & 64. \\
            65. & 66. & 67. & 68. & 69. \\
            70. & 71. & 72. & 73. & 74. \\
            75. & 76. & -1. & 78. & 79.
        \end{matrix}\right]
        \left[\begin{matrix}
            80. & 81. & 82. & 83. & 84. \\
            85. & 86. & 87. & 88. & 89. \\
            90. & 91. & 92. & -5. & 94. \\
            95. & 96. & 97. & 98. & 99.
        \end{matrix}\right]
        \left[\begin{matrix}
            100. & 101. & 102. & 103. & 104. \\
            105. &  -3. & 107. & 108. & 109. \\
            110. & 111. & 112. & 113. & 114. \\
            115. & 116. & 117. & 118. & 119.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 含义：参考 [Onnx ScatterND 算子](https://github.com/onnx/onnx/blob/main/docs/Operators.md#scatternd) 和 [TensorRT C++ API 说明](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_scatter_layer.html)
+ 数据张量 data 形状 $[d_{0},d_{1},...,d_{r-1}]$（r 维），索引张量 index 形状 $[a_{0},a_{1},...,a_{q-1}]$（q 维），更新张量 update 形状 $[b_{0},b_{1},...,b_{s-1}]$（$s = r - a_{q-1} + q - 1$ 维），输出张量 output 形状与 data 相同
+ 若 $r=a_{q-1}$（范例代码 1），则 $s=q-1$，此时对于 $0 \le i < q$，有 $b_{i} = a_{i}$（update 比 index 少一维，且各维尺寸跟 index 去掉最低维后对应相等）
+ 若 $r>a_{q-1}$（范例代码 2），则 $s>q-1$，此时对于 $0 \le i < q$，有 $b_{i} = a_{i}$，对于 $q \le i < s$，有 $b_{i} = d_{a_{q-1}+i-q}$（也即 $b_{q} = d_{a_{q-1}}, b_{q+1} = d_{a_{q-1}+1,...}$，最后一项 $i=s-1$，此时 $b_{s-1} = d_{a_{q-1}+s-1-q} = d_{r-1}$，恰好取到 data 的最后一维的尺寸）
+ 用 numpy 语法，记 
```python
q = len(index.shape)
nIndex = np.prod(index.shape[:-1])
index2D = index.reshape(nIndex,index.shape[-1])
update2D = update.reshape(nIndex,*update.shape[q-1:])
```
+ 那么计算结果可以表示为
```python
for i in nIndex:
    output[*index2D[i]] = update2D[i]
```
+ 对于上面的范例代码 1，就是：
$$
i={\color{#0000FF}{0}} \Rightarrow 
    index2D[{\color{#0000FF}{0}}] = [{\color{#FF0000}{0,2,1,1}}] \Rightarrow 
    output[{\color{#FF0000}{0,2,1,1}}] = update2D[{\color{#0000FF}{0}}] = -0. \\
    \cdots \\
i={\color{#0000FF}{5}} \Rightarrow 
    index2D[{\color{#0000FF}{5}}] = [{\color{#007F00}{1,1,2,3}}] \Rightarrow 
    output[{\color{#007F00}{1,1,2,3}}] = update2D[{\color{#0000FF}{5}}] = -5.
$$
+ 或者还原回 index 和 update 的原始下标来表示，就是：
$$
i={\color{#0000FF}{0}},j={\color{#FF7F00}{0}} \Rightarrow 
    index[{\color{#0000FF}{0}},{\color{#FF7F00}{0}}] = [{\color{#FF0000}{0,2,1,1}}] \Rightarrow 
    output[{\color{#FF0000}{0,2,1,1}}] = update[{\color{#0000FF}{0}},{\color{#FF7F00}{0}}] = -0. \\
i={\color{#0000FF}{0}},j={\color{#FF7F00}{1}} \Rightarrow 
    index[{\color{#0000FF}{0}},{\color{#FF7F00}{1}}] = [{\color{#FF0000}{1,0,3,2}}] \Rightarrow 
    output[{\color{#FF0000}{1,0,3,2}}] = update[{\color{#0000FF}{0}},{\color{#FF7F00}{1}}] = -1. \\
i={\color{#0000FF}{0}},j={\color{#FF7F00}{2}} \Rightarrow 
    index[{\color{#0000FF}{0}},{\color{#FF7F00}{2}}] = [{\color{#FF0000}{0,1,2,3}}] \Rightarrow 
    output[{\color{#FF0000}{0,1,2,3}}] = update[{\color{#0000FF}{0}},{\color{#FF7F00}{2}}] = -2. \\
\cdots \\
i={\color{#0000FF}{1}},j={\color{#FF7F00}{0}} \Rightarrow 
    index[{\color{#0000FF}{1}},{\color{#FF7F00}{0}}] = [{\color{#007F00}{1,2,1,1}}] \Rightarrow 
    output[{\color{#007F00}{1,2,1,1}}] = update[{\color{#0000FF}{1}},{\color{#FF7F00}{0}}] = -3. \\
i={\color{#0000FF}{1}},j={\color{#FF7F00}{1}} \Rightarrow 
    index[{\color{#0000FF}{1}},{\color{#FF7F00}{1}}] = [{\color{#007F00}{0,0,3,2}}] \Rightarrow 
    output[{\color{#007F00}{0,0,3,2}}] = update[{\color{#0000FF}{1}},{\color{#FF7F00}{1}}] = -4. \\
i={\color{#0000FF}{1}},j={\color{#FF7F00}{2}} \Rightarrow 
    index[{\color{#0000FF}{1}},{\color{#FF7F00}{2}}] = [{\color{#007F00}{1,1,2,3}}] \Rightarrow 
    output[{\color{#007F00}{1,1,2,3}}] = update[{\color{#0000FF}{1}},{\color{#FF7F00}{2}}] = -5.
$$
+ 对于上面的范例代码 2，就是：
$$
\begin{aligned}
    i&={\color{#0000FF}{0}} \Rightarrow 
        index2D[{\color{#0000FF}{0}}] = [{\color{#FF0000}{0,2,1}}] \Rightarrow 
        output[{\color{#FF0000}{0,2,1}}] = update2D[{\color{#0000FF}{0}}] = [-0., -1., -2., -3., -4.] \\
    &\cdots \\
    i&={\color{#0000FF}{5}} \Rightarrow 
        index2D[{\color{#0000FF}{5}}] = [{\color{#007F00}{1,1,2}}] \Rightarrow 
        output[{\color{#007F00}{1,1,2}}] = update2D[{\color{#0000FF}{5}}] = [-25., -26., -27., -28., -29.,]
\end{aligned}
$$

+ 说明：
    - 记 $nIndex = a_{0}*a_{1}*...*a_{q-2}$，
    - 把 index 变形为 $nIndex$ 行 $a_{q-1}$ 列的矩阵 index2D，用其每一行来索引 data，同时把 update 变形为 nIndex 组形状为 $[b_{q-1},...,b_{s-1}]$ 的张量 update2D
    - 如果 $r = a_{q-1}$（范例代码 1），那么 index 的第 $i$ 行作为索引恰好取到 output（或 data） 的一个元素（np.shape(output[*index2D[i]]==[])）；而此时 $b_{q-1} = b_{s-1} = 1$（因为 update 只有 $q-1$ 维，全在 $nIndex$ 维度上了），该索引在 update2D中也索引到一个元素，于是使用 update2D 的该元素来替换 output 的对应元素
    - 如果 $r > a_{q-1}$（范例代码 2），记 $nD = r - a_{q-1}$，那么 index 的第 $i$ 行作为索引会取到 output（或 data） 的一个 nD 维子张量（len(np.shape(output[*index2D[i]]))==nD），形状 $[d_{a_{q-1}},...d_{r-1}]$；此时该索引在 update2D 中也索引到一个 nD 维的子张量，形状也是 $[d_{a_{q-1}},...d_{r-1}]$，于是使用 update2D 的该元素来替换 output 的对应元素

+ 不满足 $s = r - a_{q-1} + q - 1$ 时报错：
```
[TRT] [E] 4: [graphShapeAnalyzer.cpp::computeOutputExtents::1032] Error Code 4: Miscellaneous ((Unnamed Layer* 0) [Scatter]: error while lowering shape of node)
```

+ $b_{i}$ 不满足约束条件时报错：
```
[TRT] [E] 4: [graphShapeAnalyzer.cpp::processCheck::581] Error Code 4: Internal Error ((Unnamed Layer* 0) [Scatter]: dimensions not compatible for ScatterND)
```
