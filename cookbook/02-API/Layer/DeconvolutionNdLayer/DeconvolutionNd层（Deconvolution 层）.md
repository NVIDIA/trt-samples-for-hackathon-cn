# DeonvoutionNd 层（Deconvolution 层
+ **括号中的层名和参数名于 TensorRT 8.0 被标记为弃用，TensorRT 9.0 中被移除****
+ 初始示例代码
+ num_output_maps & kernel_size_nd (kernel_size) & kernel & bias
+ stride_nd (stride)
+ padding_nd (padding)
+ pre_padding
+ post_padding
+ padding_mode
+ num_groups
+ 三维反卷积的示例
+ set_input + INT8-QDQ 模式

---
### 初始示例代码
+ 见 SimpleUsage.py

+ 输入张量形状 (1,1,3,3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 2. & 3. \\
            2. & 5. & 6. \\
            3. & 8. & 9.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (1,1,5,5)，默认反卷积步长 1，跨步 1，没有边缘填充
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \textcolor[rgb]{0,0.5,0}{10000.0000}& 21000.0000& 32100.0000& 3200.0000&  300.0000\\
            40010.0000& 54021.0000& 65432.1000& 6503.2   &  600.3000\\
            70040.0100& 87054.0210& \textcolor[rgb]{0,0,1}{98765.4321}& 9806.5032&  900.6003 \\
               70.0400&    87.0540&    98.76540&   9.8065&    0.9006\\
                0.0700&     0.0870&     0.09867&   0.0098&    \textcolor[rgb]{1,0,0}{0.0009}
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：反卷积结果中各元素的**个位**代表得出该值时卷积窗口的中心位置，其他各位代表参与计算的周围元素，**注意反卷积核是倒序的**。受限于 float32 精度，运行结果无法完整展示 9 位有效数字，以上结果矩阵手工调整了这部分显示，以展示理想运行结果。后续各参数讨论中的输出矩阵不再作调整，而是显示再有舍入误差的原始结果。
$$
\left[\begin{matrix}
    \begin{matrix}{\boxed{
        \begin{matrix} \ & \ & \ \\ \ & \  & \ \\ \ & \ & 1 \end{matrix}
    }}\end{matrix}\
    \begin{matrix} & \\ & \\ 2 & 3\end{matrix}\\
    \begin{matrix} \ \ \ \ \ \ \, & \  & 4 & 5 & 6 & \\ \ \ \ \ & & 7 & 8 & 9\end{matrix}
\end{matrix}\right]
\odot
{\boxed{\begin{matrix}
    10^{-4} & 10^{-3} & 10^{-2} \\ 10^{-1} & 1 & 10^{1} \\ 10^{2} & 10^{3} & 10^{4}
\end{matrix}}}
= \textcolor[rgb]{0,0.5,0}{10000.},
\\
\left[\quad\begin{matrix}\\
    \begin{matrix}{\boxed{
        \begin{matrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{matrix}
    }}\end{matrix}\\
    \begin{matrix}\ \end{matrix}
\end{matrix}\quad\right]
\odot
{\boxed{\begin{matrix}
    10^{-4} & 10^{-3} & 10^{-2} \\ 10^{-1} & 1 & 10^{1} \\ 10^{2} & 10^{3} & 10^{4}
\end{matrix}}}
=\textcolor[rgb]{0,0,1}{98765.4321},
\\
\left[\quad\begin{matrix}
    \begin{matrix}
    	1 & 2 & 3 & \ & \ \ \\ 4 & 5 & 6 & \ & \ \
   	\end{matrix}\\
   	\begin{matrix}
    	7 & 8 \\ \ & \ \\ & \
   	\end{matrix}\
   	{\boxed{
        \begin{matrix} \ 9 & \ & \ \\ \ & \ & \ \\ \ & \ & \ \end{matrix}
    }}
\end{matrix}\quad\right]
\odot
{\boxed{\begin{matrix}
    10^{-4} & 10^{-3} & 10^{-2} \\ 10^{-1} & 1 & 10^{1} \\ 10^{2} & 10^{3} & 10^{4}
\end{matrix}}}
=\textcolor[rgb]{1,0,0}{0.0009}
$$

+ 使用旧版 API `add_deconvolution` 会收到警告
```
DeprecationWarning: Use add_deconvolution_nd instead.
```

+ 不设置 config.set_memory_pool_limit 会收到报错：
```
[TensorRT] INFO: Some tactics do not have sufficient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.
[TensorRT] ERROR: 10: [optimizer.cpp::computeCosts::1855] Error Code 10: Internal Error (Could not find any implementation for node (Unnamed Layer* 0) [Deconvolution].)
```

+ 输入输出张量、权重尺寸计算见 [link](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#deconvolution-layer)

+ Dynamic Shape 模式下，C 维尺寸必须是构建期常量，不可为 -1

---
### num_output_maps & kernel_size_nd (kernel_size) & kernel & bias
+ 见 Num_output_Maps+Kernel_size_nd+Kernel+Bias.py，在构建 Convolution 层后再修改其输出通道数、卷积窗口尺寸、卷积核权重、偏置权重

+ 输出张量形状 (1,1,5,5)，结果与初始示例代码相同

+ 使用旧版 API `kernel_size` 会收到警告
```
DeprecationWarning: Use kernel_size_nd instead.
```

---
### stride_nd (stride)
+ 见 Stride_nd.py，设置反卷积窗口的移动步长

+ 指定 stride_nd=(2,2)（HW 维跨步均为 2），输出张量形状 (1,1,7,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1.        & 1000.    & 20100.    & 2000.    & 30200.        & 3000.    & 300.         \\
           1.     &    1.    &    20.1   &    2.    &    30.2       &    3.    &   0.3        \\
        40000.01  & 4000.001 & 50400.02  & 5000.002 & 60500.03      & 6000.003 & 600.0003     \\
           1.     &    4.    &    50.4   &    5.    &    60.5       &    6.    &   0.6        \\
        70000.04  & 7000.004 & 80700.05  & 8000.005 & 90800.06      & 9000.006 & 900.0006     \\
           1.     &    7.    &    80.7   &    8.    &    90.8       &    9.    &   0.90000004 \\
            0.07  &    0.007 &     0.0807&    0.008 &     0.09079999&    0.009 &   0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 stride_nd=(2,1)（H 维跨步为 2），输出张量形状 (1,1,7,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1.        & 21000.    & 32100.        & 3200.    &  300.         \\
           1.     &    21.    &    32.1       &    3.2   &    0.3        \\
        40000.01  & 54000.02  & 65400.035     & 6500.0034&  600.0003     \\
           1.     &    54.         65.4       &    6.5   &    0.6        \\
        70000.04  & 87000.055 & 98700.07      & 9800.007 &  900.0006     \\
           1.     &    87.    &    98.7       &    9.8   &    0.90000004 \\
            0.07  &     0.087 &     0.09869999&    0.0098&    0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 stride_nd=(1,2)（H 维跨步为 2），输出张量形状 (1,1,5,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1.        & 1000. &    20100.    & 2000.    & 30200.        & 3000.    & 300.     \\
        2.        & 4001. &    50420.1   & 5002.    & 60530.2       & 6003.    & 600.3    \\
        70040.01  & 7004.001 & 80750.42  & 8005.002 & 90860.53      & 9006.003 & 900.6003 \\
           70.04  &    7.004 &    80.7504&    8.005 &    90.860504  &    9.006 &   0.9006 \\
            0.07  &    0.007 &     0.0807&    0.008 &     0.09079999&    0.009 &   0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用旧版 API `stride` 会收到警告
```
DeprecationWarning: Use stride_nd instead
```

---
### padding_nd (padding)
+ 见 Padding_nd.py，设置反卷积输入的光环元素宽度

+ 指定 padding_nd=(1,1)（HW 维均减少填充 1 层 0），输出张量形状 (1,1,3,3)
+ 含义是给输入张量 HW 维均填充一层 0（[3,3] ->[5,5]）后做反卷积
$$
\left[\begin{matrix}
    \left[\begin{matrix}
            \left[\begin{matrix}
            1.        & 65432.1     &  6503.2  \\
            87054.02  & 98765.43    & 9806.503 \\
               87.054 &    98.765396& 9.8064995
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_nd=(1,0)（H 维减少填充 1 层 0），输出张量形状 (1,1,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.        & 54021.    & 65432.1     & 6503.2      &   600.3    \\
            70040.01  & 87054.02  & 98765.43    & 9806.503    & 900.6003   \\
               70.04  &    87.054 &    98.765396&    9.8064995&     0.9006
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_nd=(0,1)（W 维减少填充 1 层 0），输出张量形状 (1,1,5,3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.        & 32100.        & 3200.        \\
            2.        & 65432.1       & 6503.2       \\
            87054.02  & 98765.43      & 9806.503     \\
               87.054 &    98.765396  &    9.8064995 \\
                0.087 &     0.09869999&    0.0098
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_nd=(2,2)（HW 维均减少填充 2 层 0），输出张量形状 (1,1,1,1)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                98765.43
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用旧版 API `padding` 会收到警告
```
DeprecationWarning: Use padding_nd instead
```

---
### pre_padding
+ 见 Pre_padding.py，设置反卷积输入的前导光环元素宽度

+ 指定 pre_padding=(1,1)（HW 维头部均减少填充 1 层 0），输出张量形状 (1,1,4,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.        & 65432.1       &  6503.2      & 600.3    \\
            87054.02  & 98765.43      &  9806.503    & 900.6003 \\
               87.054 &    98.765396  &     9.8064995&   0.9006 \\
                0.087 &     0.09869999&     0.0098   &   0.0009
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 pre_padding=(1,0)（H 维头部减少填充 1 层 0），输出张量形状 (1,1,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.        & 54021.    & 65432.1      &  6503.2      & 600.3    \\
            70040.01  & 87054.02  & 98765.43     &  9806.503    & 900.6003 \\
               70.04  &    87.054 &    98.765396 &     9.8064995&   0.9006 \\
                0.07  &     0.087 &    0.09869999&     0.0098   &   0.0009
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 pre_padding=(0,1)（w 维头部减少填充 1 层 0），输出张量形状 (1,1,5,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.        & 32100.        & 3200.       & 300.     \\
            2.        & 65432.1       & 6503.2      & 600.3    \\
            87054.02  & 98765.43      & 9806.503    & 900.6003 \\
               87.054 &    98.765396  &    9.8064995&   0.9006 \\
                0.087 &     0.09869999&    0.0098   &   0.0009
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### post_padding
+ 见 Post_padding.py，设置反卷积输入的后置光环元素宽度

+ 指定 post_padding=(1,1)（HW 维尾部均减少填充 1 层 0），输出张量形状 (1,1,4,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1.        & 21000.    & 32100.      & 3200.        \\
        2.        & 54021.    & 65432.1     & 6503.2       \\
        70040.01  & 87054.02  & 98765.43    & 9806.503     \\
           70.04  &    87.054 &    98.765396&    9.8064995
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 post_padding=(1,0)（H 维尾部减少填充 1 层 0），输出张量形状 (1,1,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.        & 21000.    & 32100.      & 3200.       & 300.     \\
            2.        & 54021.    & 65432.1     & 6503.2      & 600.3    \\
            70040.01  & 87054.02  & 98765.43    & 9806.503    & 900.6003 \\
               70.04  &    87.054 &    98.765396&    9.8064995&   0.9006
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 post_padding=(0,1)（W 维尾部减少填充 1 层 0），输出张量形状 (1,1,5,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.        & 21000.    & 32100.        & 3200.        \\
            2.        & 54021.    & 65432.1       & 6503.2       \\
            70040.01  & 87054.02  & 98765.43      & 9806.503     \\
               70.04  &    87.054 &    98.765396  &    9.8064995 \\
                0.07  &     0.087 &     0.09869999&    0.0098
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### padding_mode
+ 见 padding_mode.py，设置反卷积输入的光环元素添加方式

+ 计算过程参考 [TensorRT C++ API reference](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#a72f43f32e90e4ac5548f8c9ae007584c)

+ 指定 padding_mode = **trt.PaddingMode.SAME_UPPER**，输出张量形状 (1,1,6,6)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.       & 1000.    & 20100.   & 2000.    & 30200.   & 3000.    \\
               1.    &    1.    &    20.1  &    2.    &    30.2     & 3.    \\
            40000.01 & 4000.001 & 50400.02 & 5000.002 & 60500.03 & 6000.003 \\
               1.    &    4.    &    50.4  &    5.    &    60.5     & 6.    \\
            70000.04 & 7000.004 & 80700.05 & 8000.005 & 90800.06 & 9000.006 \\
               1.    &     7.   &    80.7  &    8.    &    90.8     & 9.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.SAME_LOWER**，输出张量形状 (1,1,6,6)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}s
               1.    &    20.1    &    2.    &    30.2        &    3.    &   0.3        \\
            4000.001 & 50400.02   & 5000.002 & 60500.03       & 6000.003 & 600.0003     \\
               1.    &    50.4    &    5.    &    60.5        &    6.    &   0.6        \\
            7000.004 & 80700.05   & 8000.005 & 90800.06       & 9000.006 & 900.0006     \\
               1.    &    80.7    &    8.    &    90.8        &    9.    &   0.90000004 \\
               0.007 &     0.0807 &    0.008 &     0.09079999 &    0.009 &   0.0009
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.EXPLICIT_ROUND_UP**，输出张量形状 (1,1,7,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.       & 1000.    & 20100.     & 2000.    & 30200.         & 3000.    & 300.       \\
               1.    &    1.    &    20.1    &    2.    &    30.2        & 3.       & 0.3        \\
            40000.01 & 4000.001 & 50400.02   & 5000.002 & 60500.03       & 6000.003 & 600.0003   \\
               1.    &    4.    &    50.4    &    5.    &    60.5        & 6.       & 0.6        \\
            70000.04 & 7000.004 & 80700.05   & 8000.005 & 90800.06       & 9000.006 & 900.0006   \\
               1.    &    7.    &    80.7    &    8.    &    90.8        & 9.       & 0.90000004 \\
                0.07 &    0.007 &     0.0807 &    0.008 &     0.09079999 & 0.009    & 0.0009
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.EXPLICIT_ROUND_DOWN**，输出张量形状 (1,1,7,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.       & 1000.    & 20100.     & 2000.    & 30200.         & 3000.    & 300.       \\
               1.    &    1.    &    20.1    &    2.    &    30.2        &    3.    & 0.3        \\
            40000.01 & 4000.001 & 50400.02   & 5000.002 & 60500.03       & 6000.003 & 600.0003   \\
               1.    &    4.    &    50.4    &    5.    &    60.5        &    6.    & 0.6        \\
            70000.04 & 7000.004 & 80700.05   & 8000.005 & 90800.06       & 9000.006 & 900.0006   \\
               1.    &    7.    &    80.7    &    8.    &    90.8        &    9.    & 0.90000004 \\
                0.07 &    0.007 &     0.0807 &    0.008 &     0.09079999 &    0.009 & 0.0009
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.CAFFE_ROUND_UP**，输出张量形状 (1,1,7,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.       & 1000.    & 20100.     & 2000.    & 30200.         & 3000.    & 300.         \\
               1.    &    1.    &    20.1    &    2.    &    30.2        &    3.    &   0.3        \\
            40000.01 & 4000.001 & 50400.02   & 5000.002 & 60500.03       & 6000.003 & 600.0003     \\
               1.    &    4.    &    50.4    &    5.    &    60.5        &    6.    &   0.6        \\
            70000.04 & 7000.004 & 80700.05   & 8000.005 & 90800.06       & 9000.006 & 900.0006     \\
               1.    &    7.    &    80.7    &    8.    &    90.8        &    9.    &   0.90000004 \\
                0.07 &    0.007 &     0.0807 &    0.008 &     0.09079999 &    0.009 &   0.0009
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.CAFFE_ROUND_DOWN**，输出张量形状 (1,1,7,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.       & 1000.    & 20100.     & 2000.    & 30200.         & 3000.    & 300.         \\
               1.    &    1.    &    20.1    &    2.    &    30.2        &    3.    &   0.3        \\
            40000.01 & 4000.001 & 50400.02   & 5000.002 & 60500.03       & 6000.003 & 600.0003     \\
               1.    &    4.    &    50.4    &    5.    &    60.5        &    6.    &   0.6        \\
            70000.04 & 7000.004 & 80700.05   & 8000.005 & 90800.06       & 9000.006 & 900.0006     \\
               1.    &    7.    &    80.7    &    8.    &    90.8        &    9.    &   0.90000004 \\
                0.07 &    0.007 &     0.0807 &    0.008 &     0.09079999 &    0.009 &   0.0009
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### num_groups
+ 见 Num_groups.py，分组反卷积的分组数

+ 指定 num_groupds=2，输入张量和卷积核均在 C 维上被均分为 2 组，各自卷积后再拼接到一起，输出张量形状 (1,2,4,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.       & 21000.    & 32100.         & 3200.        & 300.     \\
            2.       & 54021.    & 65432.1        & 6503.2       & 600.3    \\
            70040.01 & 87054.02  & 98765.43       & 9806.503     & 900.6003 \\
               70.04 &    87.054 &    98.76539    &    9.8064995 & 0.9006   \\
                0.07 &     0.087 &     0.09869999 &    0.0098    & 0.0009
        \end{matrix}\right] \\
        \left[\begin{matrix}
            -10000.   & -21000.     & -32100.         & -3200.     & -300.     \\
            -40010.   & -54021.     & -65432.1        & -6503.2    & -600.3    \\
            -70040.01 & -87054.02   & -98765.43       & -9806.503  & -900.6003 \\
               -70.04 &     -87.054 &    -98.76539    & -9.8064995 & -0.9006   \\
                -0.07 &     -0.087  &     -0.09869999 & -0.0098    & -0.0009
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ int8 模式中，每组的尺寸（nC/nGroup 和 nCOut/nGroup）必须是 4 的倍数

---
### 三维反卷积的示例
+ 见 Deconvolution3D.py，使用三维反卷积

$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                10000.   & 21000.    & 32100.         & 3200.        & 300.     \\
                40010.   & 54021.    & 65432.1        & 6503.2       & 600.3    \\
                70040.01 & 87054.02  & 98765.43       & 9806.503     & 900.6003 \\
                   70.04 &    87.054 &    98.76539    &    9.8064995 &   0.9006 \\
                    0.07 &     0.087 &     0.09869999 &    0.0098    &   0.0009
            \end{matrix}\right] \\
            \left[\begin{matrix}
                0. & 0. & 0. & 0. & 0. \\
                0. & 0. & 0. & 0. & 0. \\
                0. & 0.00099945 & 0.0019989 & 0.00007031 & 0. \\
                0. & 0.00000525 & 0.        & 0.00000014 & 0. \\
                0. & 0. & 0. & 0. & 0.
            \end{matrix}\right] \\
            \left[\begin{matrix}
                -10000.   & -21000.    & -32100.         & -3200.        & -300.     \\
                -40010.   & -54021.    & -65432.1        & -6503.2       & -600.3    \\
                -70040.01 & -87054.02  & -98765.43       & -9806.503     & -900.6003 \\
                   -70.04 &    -87.054 &    -98.76539    &    -9.8064995 &   -0.9006 \\
                    -0.07 &     -0.087 &     -0.09869999 &    -0.0098    &   -0.0009
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### set_input + INT8-QDQ 模式
+ 见 Set_input+INT8QDQ.py，使用 set_input 接口和 INT8 QDQ 模式

+ 参考 [link](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_deconvolution_layer.html#aa1eb8deb3192489196cb7884a4177de4)

+ 输出张量形状 (1,1,5,5)

$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            127. &  381. &  735. &  581. &  300. \\
            518. & 1164. & 1829. & 1265. &  600. \\
            929. & 1959. & 2924. & 1949. &  900. \\
              0.   &    0. &    0. &    0. &  0.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$
