# ConvoutionNd 层（Convolution 层）
+ **括号中的层名和参数名于 TensorRT 8.0 被标记为弃用，TensorRT 9.0 中被移除**
+ 初始范例代码
+ num_output_maps & kernel_size_nd (kernel_size) & kernel & bias
+ stride_nd (stride)
+ padding_nd (padding)
+ pre_padding
+ post_padding
+ padding_mode
+ dilation_nd (dilation)
+ num_groups
+ 三维卷积的范例
+ set_input + INT8-QDQ 模式

---
### 初始范例代码
+ 见 SimpleUsage.py

+ 输入张量形状 (1,1,6,9)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 2. & 3. & 1. & 2. & 3. & 1. & 2. & 3. \\
            2. & 5. & 6. & 4. & 5. & 6. & 4. & 5. & 6. \\
            3. & 8. & 9. & 7. & 8. & 9. & 7. & 8. & 9. \\
            4. & 2. & 3. & 1. & 2. & 3. & 1. & 2. & 3. \\
            5. & 5. & 6. & 4. & 5. & 6. & 4. & 5. & 6. \\
            6. & 8. & 9. & 7. & 8. & 9. & 7. & 8. & 9. \\
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (1,1,4,7)，默认卷积步长 1，跨步 1，没有边缘填充
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \textcolor[rgb]{0,0.5,0}{12345.6789} & \textcolor[rgb]{0,0,1}{23156.4897}  & 31264.5978 & 12345.6789 & 23156.4897  & 31264.5978 & 12345.6789 \\
            45678.9123 & 56489.7231 & 64597.8312 & 45678.9123 & 56489.7231 & 64597.8312 & 45678.9123 \\
            78912.3456  & 89723.1564  & 97831.2645  & 78912.3456  & 89723.1564  & 97831.2645  & 78912.3456  \\
            12345.6789 & 23156.4897  & 31264.5978 & 12345.6789 & 23156.4897  & 31264.5978 & 12345.6789
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：卷积结果中各元素的**个位**代表得出该值时卷积窗口的中心位置，其他各位代表参与计算的周围元素。受限于 float32 精度，运行结果无法完整展示 9 位有效数字，以上结果矩阵手工调整了这部分显示，以展示理想运行结果。后续各参数讨论中的输出矩阵不再作调整，而是显示再有舍入误差的原始结果。
$$
\left[\quad\begin{matrix}
    \begin{matrix}{\boxed{
        \begin{matrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{matrix}
    }}\end{matrix}\
    \begin{matrix} 1 & \cdots \\ 4 \\ 7 \end{matrix}\\
    \begin{matrix} \ \, 1 & 2 & 3 & 1 & \cdots \\ \ \,\vdots & & & \vdots \end{matrix}
\end{matrix}\right]
\odot
{\boxed{\begin{matrix}
    10^{4} & 10^{3} & 10^{2} \\ 10^{1} & 1 & 10^{-1} \\ 10^{-2} & 10^{-3} & 10^{-4}
\end{matrix}}}
= \textcolor[rgb]{0,0.5,0}{12345.6789},
\\
\left[\quad\begin{matrix}
    \begin{matrix} 1 \\ 4 \\ 7 \end{matrix}\
    \begin{matrix}{\boxed{
        \begin{matrix}2 & 3 & 1 \\ 5 & 6 & 4 \\ 8 & 9 & 7\end{matrix}
    }}\end{matrix}\
    \begin{matrix} \cdots \\ \\ \\ \end{matrix}\\
    \begin{matrix} 1 & 2 & 3 & 1 & \cdots \\ \vdots & & & \vdots \end{matrix}
\end{matrix}\right]
\odot
{\boxed{\begin{matrix}
    10^{4} & 10^{3} & 10^{2} \\ 10^{1} & 1 & 10^{-1} \\ 10^{-2} & 10^{-3} & 10^{-4}
\end{matrix}}}
=\textcolor[rgb]{0,0,1}{23156.4897}
$$

+ 使用旧版 API `add_convolution` 会收到警告：
```
DeprecationWarning: Use add_convolution_nd instead.
```

+ 输入输出张量、权重尺寸计算见 [link](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#convolution-layer)

+ Dynamic Shape 模式下，C 维尺寸必须是构建期常量，不可为 -1

---
### num_output_maps & kernel_size_nd (kernel_size) & kernel & bias
+ 见 Num_output_Maps+Kernel_size_nd+Kernel+Bias.py，在构建 Convolution 层后再修改其输出通道数、卷积窗口尺寸、卷积核权重、偏置权重

+ 输出张量形状 (1,1,4,7)，结果与初始范例代码相同

+ 使用旧版 API `kernel_size` 会收到警告
```
DeprecationWarning: Use kernel_size_nd instead.
```

---
### stride_nd (stride)
+ 见 Stride_nd.py，设置卷积窗口的移动步长

+ 指定 stride_nd=(2,2)（HW 维跨步均为 2），输出张量形状 (1,1,2,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            12345.679 & 31264.598 & 23156.49 & 12345.679 \\
            78912.34  & 97831.27  & 89723.16 & 78912.34
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 stride_nd=(2,1)（H 维跨步 2），输出张量形状 (1,2,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            12345.679 & 23156.49 & 31264.598 & 12345.679 & 23156.49 & 31264.598 & 12345.679 \\
            78912.34  & 89723.16 & 97831.27  & 78912.34  & 89723.16 & 97831.27  & 78912.34
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 stride_nd=(1,2)（W 维跨步 2），输出张量形状 (1,4,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            12345.679 & 31264.598 & 23156.49  & 12345.679 \\
            45678.914 & 64597.832 & 56489.723 & 45678.914 \\
            78912.34  & 97831.27  & 89723.16  & 78912.34  \\
            12345.679 & 31264.598 & 23156.49  & 12345.679
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用旧版 API `stride` 会收到警告
```
DeprecationWarning: Use stride_nd instead
```

---
### padding_nd (padding)
+ 见 Padding_nd.py，设置卷积输入的光环元素宽度

+ 指定 padding_nd=(1,1)（HW 维均填充 1 层 0），输出张量形状 (1,1,6,9)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
               1.2045&    12.3456&    23.1564&    31.2645&    12.3456&    23.1564&    31.2645&    12.3456&    23.056 \\
            1204.5078& 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09  \\
            4507.801 & 45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 & 56089.023 \\
            7801.2046& 78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  & 89023.055 \\
            1204.5078& 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09  \\
            4507.8   & 45678.9   & 56489.7   & 64597.8   & 45678.9   & 56489.7   & 64597.8   & 45678.9   & 56089.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_nd=(1,0)（H 维填充 1 层 0），输出张量形状 (1,1,6,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
               12.3456&    23.1564&    31.2645&    12.3456&    23.1564&    31.2645&    12.3456\\
            12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
            45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 \\
            78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  \\
            12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
            45678.9   & 56489.7   & 64597.8   & 45678.9   & 56489.7   & 64597.8   & 45678.9   \\
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_nd=(0,1)（W 维填充 1 层 0），输出张量形状 (1,1,4,9)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1204.5078 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09  \\
            4507.801  & 45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 & 56089.023 \\
            7801.2046 & 78912.34  &  89723.16 &  97831.27 & 78912.34  & 89723.16  & 97831.27  & 78912.34  & 89023.055 \\
            1204.5078 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09
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
+ 见 Pre_padding.py，设置卷积输入的前导光环元素宽度

+ 指定 pre_padding=(1,1)（HW 维头部均填充 1 层 0），输出张量形状 (1,1,5,8)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
               1.2045&    12.3456&     23.1564&    31.2645&    12.3456&    23.1564&    31.2645&   12.3456 \\
            1204.5078& 12345.679 &  23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
            4507.801 & 45678.914 &  56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 \\
            7801.2046& 78912.34  &  89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  \\
            1204.5078& 12345.679 &  23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 pre_padding=(1,0)（H 维头部填充 1 层 0），输出张量形状 (1,1,5,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
               12.3456&    23.1564&    31.2645&    12.3456&    23.1564&    31.2645&    12.3456\\
            12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
            45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 \\
            78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  \\
            12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 pre_padding=(0,1)（W 维头部填充 1 层 0），输出张量形状 (1,1,4,8)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1204.5078& 12345.679&  23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
            4507.801 & 45678.914&  56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 \\
            7801.2046& 78912.34 &  89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  \\
            1204.5078& 12345.679&  23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### post_padding
+ 见 Post_padding.py，设置卷积输入的后置光环元素宽度

+ 指定 post_padding=(1,1)（HW 维尾部均填充 1 层 0），输出张量形状 (1,1,5,8)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09  \\
            45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 & 56089.023 \\
            78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  & 89023.055 \\
            12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09  \\
            45678.9   & 56489.7   & 64597.8   & 45678.9   & 56489.7   & 64597.8   & 45678.9   & 56089.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 post_padding=(1,0)（H 维尾部填充 1 层 0），输出张量形状 (1,1,5,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
            45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 \\
            78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  \\
            12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
            45678.9   & 56489.7   & 64597.8   & 45678.9   & 56489.7   & 64597.8   & 45678.9
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 post_padding=(0,1)（W 维尾部填充 1 层 0），输出张量形状 (1,1,4,8)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09  \\
            45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 & 56089.023 \\
            78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  & 89023.055 \\
            12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### padding_mode
+ 见 Padding_mode.py，设置卷积输入的光环元素添加方式

+ 计算过程参考 [TensorRT C++ API reference](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#a72f43f32e90e4ac5548f8c9ae007584c)

+ 指定 padding_mode = **trt.PaddingMode.SAME_UPPER**，输出张量形状 (1,1,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1204.5078& 23156.49  &  12345.679 & 31264.598 & 23056.09  \\
            7801.2046& 89723.16  &  78912.34  & 97831.27  & 89023.055 \\
            4507.8   & 56489.7   &  45678.9   & 64597.8   & 56089.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.SAME_LOWER**，输出张量形状 (1,1,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.2045   & 23.1564   & 12.3456   & 31.2645   & 23.056    \\
            4507.801 & 56489.723 & 45678.914 & 64597.832 & 56089.023 \\
            1204.5078& 23156.49  & 12345.679 & 31264.598 & 23056.09
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.EXPLICIT_ROUND_UP**，输出张量形状 (1,1,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            12345.679 & 31264.598 & 23156.49 & 12345.679 \\
            78912.34  & 97831.27  & 89723.16 & 78912.34  \\
            45678.9   & 64597.8   & 56489.7  & 45678.9
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.EXPLICIT_ROUND_DOWN**，输出张量形状 (1,1,2,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            12345.679 & 31264.598 & 23156.49 & 12345.679 \\
            78912.34  & 97831.27  & 89723.16 & 78912.34
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.CAFFE_ROUND_UP**，输出张量形状 (1,1,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            12345.679 31264.598 23156.49  12345.679 \\
            78912.34  97831.27  89723.16  78912.34  \\
            45678.9   64597.8   56489.7   45678.9
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.CAFFE_ROUND_DOWN**，输出张量形状 (1,1,2,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            12345.679 & 31264.598 & 23156.49 & 12345.679 \\
            78912.34  & 97831.27  & 89723.16 & 78912.34
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### dilation_nd (dilation)
+ 见 Dilation_nd.py，设置卷积窗口的相邻元素跨度

+ 指定 dilation_nd=(2,2)（卷积核在 HW 维上元素间隔均为 2），输出张量形状 (1,1,2,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            13279.847 & 21387.955 & 32198.766 & 13279.847 & 21387.955 \\
            46513.277 & 54621.387 & 65432.2   & 46513.277 & 54621.387
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 dilation_nd=(2,1)（卷积核在 H 维上元素间隔为 2），输出张量形状 (1,1,2,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            12378.946 & 23189.756 & 31297.865 & 12378.946 & 23189.756 & 31297.865 & 12378.946 \\
            45612.38  & 56423.188 & 64531.297 & 45612.38  & 56423.188 & 64531.297 & 45612.38
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 dilation_nd=(1,2)（卷积核在 W 维上元素间隔为 2），输出张量形状 (1,1,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            13246.58  & 21354.688 & 32165.498 & 13246.58  & 21354.688 \\
            46579.816 & 54687.918 & 65498.734 & 46579.816 & 54687.918 \\
            79813.25  & 87921.35  & 98732.17  & 79813.25  & 87921.35  \\
            13246.58  & 21354.688 & 32165.498 & 13246.58  & 21354.688
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用旧版 API `dilation` 会收到警告
```
DeprecationWarning: Use dilation_nd instead
```

---
### num_groups
+ 见 Num_groups.py，分组卷积的分组数

+ 指定 num_groupds=2，输入张量和卷积核均在 C 维上被均分为 2 组，各自卷积后再拼接到一起，输出张量形状 (1,2,4,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
            45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 \\
            78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  \\
            12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679
        \end{matrix}\right] \\
        \left[\begin{matrix}
            -12345.679 & -23156.49  & -31264.598 & -12345.679 & -23156.49  & -31264.598 & -12345.679 \\
            -45678.914 & -56489.723 & -64597.832 & -45678.914 & -56489.723 & -64597.832 & -45678.914 \\
            -78912.34  & -89723.16  & -97831.27  & -78912.34  & -89723.16  & -97831.27  & -78912.34  \\
            -12345.679 & -23156.49  & -31264.598 & -12345.679 & -23156.49  & -31264.598 & -12345.679
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ int8 模式中，每组的尺寸（nC/nGroup 和 nCOut/nGroup）必须是 4 的倍数

---
### 三维卷积的范例
+ 见 Convolution3D.py，使用三维卷积

+ 输出张量形状 (1,1,1,4,7)，相当于把前面 num_groups 例子中结果的两个通道加在一起，得到了全部元素均为 0 的结果
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                -0.00018907 &  0.00053437 & -0.00014376 & -0.00018907 &  0.00053437 & -0.00014376 & -0.00018907 \\
                 0.00176249 & -0.00044376 &  0.00083124 &  0.00176249 & -0.00044376 &  0.00083124 &  0.00176249 \\
                -0.00185    & -0.00015    &  0.0089375  & -0.00185    & -0.00015    &  0.0089375  & -0.00185    \\
                -0.00018907 &  0.00053437 & -0.00014376 & -0.00018907 &  0.00053437 & -0.00014376 & -0.00018907
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### set_input + INT8-QDQ 模式
+ 见 Set_input+INT8QDQ.py，使用 set_input 接口和 INT8 QDQ 模式

+ 参考 [link](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_convolution_layer.html#aa1eb8deb3192489196cb7884a4177de4)

+ 输出张量形状 (1,1,4,7)

$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.   &  791. &  772. &  726. &  791. &  772. &  726. \\
            1.   & 1886. & 1867. & 1821. & 1886. & 1867. & 1821. \\
            2.   & 2882. & 2863. & 2817. & 2882. & 2863. & 2817. \\
            2.   &  791. &  772. &  726. &  791. &  772. &  726.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$
