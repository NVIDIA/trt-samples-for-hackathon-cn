# Scatter Layer

+ Steps to run.

```bash
python3 main.py
```

+ mode（since TensorRT 8.2）
    - Scatter ELEMENT mode
    - Scatter ND mode


### ELEMENT mode


+ Meaning: refer to [Onnx ScatterElements operator](https://github.com/onnx/onnx/blob/main/docs/Operators.md#scatterelements) and [TensorRT C++ API docs](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_scatter_layer.html).
+ Data tensor `data`, index tensor `index`, update tensor `update`, and output tensor `output` have the same shape ($dim=r$), all equal to $[d_{0},d_{1},...,d_{r-1}]$. With $axis=p$ ($0 \le p < r$):
+ Let loop variable $i_{j}$ satisfy $ 0 \le j < r, 0 \le i_{j} < d_{j}$. The computation in NumPy form is:$output[i_{0},i_{1},...,i_{p-1},index[i_{0},i_{1},...,i_{p-1},i_{p},i_{p+1},...,i_{r-1}],i_{p+1},...,i_{r-1}] = data[i_{0},i_{1},...,i_{p-1},i_{p},i_{p+1},...,i_{r-1}]$
+ For the example code above:
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

+ This formula is exactly the GatherElements formula with indices on both sides swapped.
+ Update order for output elements is not guaranteed. If two updates target the same output position with different values, the final chosen value is undefined. For example, if in the sample code `data1` is changed to ```data1 = np.zeros([nB, nC, nH, nW], dtype=np.int32)```, then `output[:,:,0,:]` can become negative random integers from `update`.

### ND mode
+ Refer to ModeND.py and ModeND2.py
+ Shape of input tensor 1: (2, 3, 4, 5)
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

+ Shape of input tensor 1: (2, 3, 4)
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

+ Shape of input tensor 1: (2, 3)
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

+ Meaning: refer to [Onnx ScatterND operator](https://github.com/onnx/onnx/blob/main/docs/Operators.md#scatternd) and [TensorRT C++ API docs](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_scatter_layer.html).
+ Data tensor `data` has shape $[d_{0},d_{1},...,d_{r-1}]$ (rank r), index tensor `index` has shape $[a_{0},a_{1},...,a_{q-1}]$ (rank q), update tensor `update` has shape $[b_{0},b_{1},...,b_{s-1}]$ (rank $s = r - a_{q-1} + q - 1$), and output tensor `output` has the same shape as `data`.
+ If $r=a_{q-1}$ (sample 1), then $s=q-1$. For $0 \le i < q$, $b_{i} = a_{i}$ (update has one less dimension than index, matching index after removing its last dimension).
+ If $r>a_{q-1}$ (sample 2), then $s>q-1$. For $0 \le i < q$, $b_{i} = a_{i}$; for $q \le i < s$, $b_{i} = d_{a_{q-1}+i-q}$ (i.e., $b_q=d_{a_{q-1}}$, ..., and at $i=s-1$, $b_{s-1}=d_{r-1}$).
+ In NumPy form, let
```python
q = len(index.shape)
nIndex = np.prod(index.shape[:-1])
index2D = index.reshape(nIndex,index.shape[-1])
update2D = update.reshape(nIndex,*update.shape[q-1:])
```
+ Then the computation can be written as
```python
for i in nIndex:
    output[*index2D[i]] = update2D[i]
```
+ For sample code 1 above:
$$
i={\color{#0000FF}{0}} \Rightarrow
    index2D[{\color{#0000FF}{0}}] = [{\color{#FF0000}{0,2,1,1}}] \Rightarrow
    output[{\color{#FF0000}{0,2,1,1}}] = update2D[{\color{#0000FF}{0}}] = -0. \\
    \cdots \\
i={\color{#0000FF}{5}} \Rightarrow
    index2D[{\color{#0000FF}{5}}] = [{\color{#007F00}{1,1,2,3}}] \Rightarrow
    output[{\color{#007F00}{1,1,2,3}}] = update2D[{\color{#0000FF}{5}}] = -5.
$$
+ Or equivalently, using original indices of index and update:
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
+ For sample code 2 above:
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

+ Notes:
    - Let $nIndex = a_{0}*a_{1}*...*a_{q-2}$.
    - Reshape `index` into matrix `index2D` with $nIndex$ rows and $a_{q-1}$ columns; each row indexes `data`. Reshape `update` into `update2D` with `nIndex` tensors of shape $[b_{q-1},...,b_{s-1}]$.
    - If $r = a_{q-1}$ (sample 1), row $i$ of `index` indexes exactly one element in `output` (or `data`) (`np.shape(output[*index2D[i]]==[])`). At this time $b_{q-1}=b_{s-1}=1$ (because update has only $q-1$ dimensions), and this index also selects one element in `update2D`, which replaces the corresponding output element.
    - If $r > a_{q-1}$ (sample 2), let $nD = r - a_{q-1}$. Row $i$ of `index` selects an nD-dimensional sub-tensor in `output` (or `data`) with shape $[d_{a_{q-1}},...d_{r-1}]$. The same index in `update2D` selects an nD-dimensional sub-tensor with the same shape, which replaces the corresponding output sub-tensor.

+ Error when $s = r - a_{q-1} + q - 1$ is not satisfied:
```
[TRT] [E] 4: [graphShapeAnalyzer.cpp::computeOutputExtents::1032] Error Code 4: Miscellaneous ((Unnamed Layer* 0) [Scatter]: error while lowering shape of node)
```

+ Error when $b_{i}$ does not satisfy constraints:
```
[TRT] [E] 4: [graphShapeAnalyzer.cpp::processCheck::581] Error Code 4: Internal Error ((Unnamed Layer* 0) [Scatter]: dimensions not compatible for ScatterND)
```
