# PoolingNd 层（Pooling 层 deprecated since TensorRT 8.0）
+ **括号中的层名和参数名于 TensorRT 8.0 被标记为弃用，TensorRT 9.0 中被移除**
+ Simple example
+ type
+ blend_factor
+ window_size_nd (window_size)
+ stride_nd (stride)
+ padding_nd (padding)
+ pre_padding
+ post_padding
+ padding_mode
+ average_count_excludes_padding
+ 三维池化的范例

+ 使用旧版 API `add_pooling` 会收到警告：
```
DeprecationWarning: Use add_pooling_nd instead.
```

---
## Simple example
+ Refer to SimpleExample.py
+ Shape of input tensor 0: (1,1,6,9)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 2. & 3. & 1. & 2. & 3. & 1. & 2. & 3. \\
            4. & 5. & 6. & 4. & 5. & 6. & 4. & 5. & 6. \\
            7. & 8. & 9. & 7. & 8. & 9. & 7. & 8. & 9. \\
            1. & 2. & 3. & 1. & 2. & 3. & 1. & 2. & 3. \\
            4. & 5. & 6. & 4. & 5. & 6. & 4. & 5. & 6. \\
            7. & 8. & 9. & 7. & 8. & 9. & 7. & 8. & 9.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (1,1,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            {\color{#007F00}{5.}} & {\color{#0000FF}{6.}} & 6. & 5. \\
            8. & 9. & 9. & 8. \\
            8. & 9. & 9. & 8.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\left[\quad\begin{matrix}
    \begin{matrix}{\boxed{
        \begin{matrix} 1 & 2 \\ 4 & 5 \end{matrix}
    }}\end{matrix}\
    \begin{matrix} 3 & \cdots \\ 6 \end{matrix}\\
    \begin{matrix} \ \, 7 & 8 & 9 & \cdots \\ \ \,\vdots & & \vdots \end{matrix}
\end{matrix}\right]
= {\color{#007F00}{5.}},

\left[\quad\begin{matrix}
    \begin{matrix} 1 \\ 4 \end{matrix}\
    \begin{matrix}{\boxed{
        \begin{matrix}2 & 3 \\ 5 & 6 \end{matrix}
    }}\end{matrix}\
    \begin{matrix} \cdots \\ \\ \end{matrix}\\
    \begin{matrix} 7 & 8 & 9 & \cdots \\ \vdots & & \vdots \end{matrix}
\end{matrix}\right]
={\color{#0000FF}{6.}}
$$

---

## type
+ Refer to Type.py，在构建 Pooling 层后再修改其池化方法

+ 指定平均值池化, shape of output tensor 0: (1,1,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.  & 2.5 & 3.  & 2.  \\
            3.5 & 4.  & 4.5 & 3.5 \\
            1.  & 5.5 & 6.  & 5.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 可用的池化方式
| trt.PoolingType 名 |                 说明                 |
| :----------------: | :----------------------------------: |
|      AVERAGE       |               均值池化               |
|        MAX         |              最大值池化              |
| MAX_AVERAGE_BLEND  | 混合池化，见下面的 blend_factor 部分 |

---

## blend_factor
+ Refer to Blend_factor.py，指定混合池化模式下两种池化方式的权重

+ 指定 blend_factor=0.5, shape of output tensor 0: (1,1,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.   & 4.75 & 5.   &   4. \\
            6.25 & 7.   & 7.25 & 6.25 \\
            1.   & 7.75 & 8.   &   7. \\
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
bF \cdot
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.  & 3.5 & 4.  & 3.  \\
            4.5 & 5.  & 5.5 & 4.5 \\
            1.  & 6.5 & 7.  & 6.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
+ \left( 1 - bF \right) \cdot
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            5. & 6. & 6. & 5. \\
            8. & 9. & 9. & 8. \\
            8. & 9. & 9. & 8.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
=
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            4.   & 4.75 & 5.   &   4. \\
            6.25 & 7.   & 7.25 & 6.25 \\
            7.   & 7.75 & 8.   &   7. \\
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## window_size_nd (window_size)
+ Refer to Window_size_nd.py，在构建 Pooling 层后再修改其池化窗口大小

+ Shape of output tensor 0: (1,1,5,8)，池化窗口尺寸不变，但是步长会保持 (1,1)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 5. & 5. & 4. & 5. & 5. & 4. & 5. \\
            2. & 8. & 8. & 7. & 8. & 8. & 7. & 8. \\
            3. & 8. & 8. & 7. & 8. & 8. & 7. & 8. \\
            4. & 5. & 5. & 4. & 5. & 5. & 4. & 5. \\
            5. & 8. & 8. & 7. & 8. & 8. & 7. & 8.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用旧版 API `window_size` 会收到警告
```
DeprecationWarning: Use window_size_nd instead.
```

---

## stride_nd (stride)
+ Refer to Stride_nd.py，指定池化窗口的移动步长

+ 指定 stride=(hS,wS), shape of output tensor 0: (1,1,5,8)，
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            5. & 6. & 6. & 5. & 6. & 6. & 5. & 6. \\
            8. & 9. & 9. & 8. & 9. & 9. & 8. & 9. \\
            8. & 9. & 9. & 8. & 9. & 9. & 8. & 9. \\
            5. & 6. & 6. & 5. & 6. & 6. & 5. & 6. \\
            8. & 9. & 9. & 8. & 9. & 9. & 8. & 9.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用旧版 API `stride` 会收到警告
```
DeprecationWarning: Use stride_nd instead.
```

---

## padding_nd (padding)
+ Refer to Padding_nd.py，设置池化输入的光环元素宽度

+ 指定 padding=(1,1)(HW 维均填充 1 层 0）, shape of output tensor 0: (1,1,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 3. & 2. & 3. & 3. \\
            2. & 9. & 8. & 9. & 9. \\
            3. & 6. & 5. & 6. & 6. \\
            4. & 9. & 8. & 9. & 9.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding=(1,0)(H 维填充 1 层 0）, shape of output tensor 0: (1,1,4,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            2. & 3. & 3. & 2. \\
            8. & 9. & 9. & 8. \\
            5. & 6. & 6. & 5. \\
            8. & 9. & 9. & 8.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding=(0,1)(W 维填充 1 层 0）, shape of output tensor 0: (1,1,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            4. & 6. & 5. & 6. & 6. \\
            7. & 9. & 8. & 9. & 9. \\
            7. & 9. & 8. & 9. & 9.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用旧版 API `padding` 会收到警告
```
DeprecationWarning: Use padding_nd instead.
```

+ 使用最大值池化并且使用 padding 时，填充数值为 fp32 或 fp16 或 int8 表示范围的最小值。比如设置 window_size_nd 和 padding_nd 均为 (2, 2) 就会看到输出张量周围一圈负值

+ 使用平均值池化并且使用 padding 时，填充数值不参与计算。比如设置 window_size_nd 和 padding_nd 均为 (2, 2) 就会收到报错信息：
```
[TRT] [E] 4:  the padding size is larger than or equal to the filter size for exclusive-counting pooling
[TRT] [E] 4: [network.cpp::validate::2917] Error Code 4: Internal Error (Layer (Unnamed Layer* 0) [Pooling] failed validation)

```

---

## pre_padding
+ Refer to Pre_padding.py，设置池化输入的前置光环元素宽度

+ 指定 pre_padding=(1,1)（HW 维头部均填充 1 层 0）, shape of output tensor 0: (1,1,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 3. & 2. & 3. & 3. \\
            7. & 9. & 8. & 9. & 9. \\
            4. & 6. & 5. & 6. & 6.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 pre_padding=(1,0)（H 维头部填充 1 层 0）, shape of output tensor 0: (1,1,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            2. & 3. & 3. & 2. \\
            8. & 9. & 9. & 8. \\
            5. & 6. & 6. & 5.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 pre_padding=(0,1)（W 维头部填充 1 层 0）, shape of output tensor 0: (1,1,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            4. & 6. & 5. & 6. & 6. \\
            7. & 9. & 8. & 9. & 9. \\
            7. & 9. & 8. & 9. & 9.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## post_padding
+ Refer to Post_padding.py，设置池化输入的后置光环元素宽度

+ 指定 post_padding=(1,1)（HW 维尾部均填充 1 层 0）, shape of output tensor 0: (1,1,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            5. & 6. & 6. & 5. & 6. \\
            8. & 9. & 9. & 8. & 9. \\
            8. & 9. & 9. & 8. & 9.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 post_padding=(1,0)（H 维尾部填充 1 层 0）, shape of output tensor 0: (1,1,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            5. & 6. & 6. & 5. \\
            8. & 9. & 9. & 8. \\
            8. & 9. & 9. & 8.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 post_padding=(0,1)（W 维尾部填充 1 层 0）, shape of output tensor 0: (1,1,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            5. & 6. & 6. & 5. & 6. \\
            8. & 9. & 9. & 8. & 9. \\
            8. & 9. & 9. & 8. & 9.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## padding_mode
+ Refer to Padding_mode.py，设置池化输入的光环元素添加方式

+ 计算过程参考 [TensorRT C++ API reference](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#a72f43f32e90e4ac5548f8c9ae007584c)

+ 指定 padding_mode = **trt.PaddingMode.SAME_UPPER**, shape of output tensor 0: (1,1,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            5. & 6. & 6. & 5. & 6. \\
            8. & 9. & 9. & 8. & 9. \\
            5. & 6. & 6. & 5. & 6.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.SAME_LOWER**, shape of output tensor 0: (1,1,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 3. & 2. & 3. & 3. \\
            7. & 9. & 8. & 9. & 9. \\
            4. & 6. & 5. & 6. & 6.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.EXPLICIT_ROUND_UP**, shape of output tensor 0: (1,1,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            5. & 6. & 6. & 5. & 6. \\
            8. & 9. & 9. & 8. & 9. \\
            5. & 6. & 6. & 5. & 6.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.EXPLICIT_ROUND_DOWN**, shape of output tensor 0: (1,1,2,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            5. & 6. & 6. & 5. \\
            8. & 9. & 9. & 8.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.CAFFE_ROUND_UP**, shape of output tensor 0: (1,1,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            5. & 6. & 6. & 5. & 6. \\
            8. & 9. & 9. & 8. & 9. \\
            5. & 6. & 6. & 5. & 6.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 padding_mode = **trt.PaddingMode.CAFFE_ROUND_DOWN**, shape of output tensor 0: (1,1,2,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            5. & 6. & 6. & 5. \\
            8. & 9. & 9. & 8.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## average_count_excludes_padding
+ Refer to Average_count_excludes_padding.py，设置在计算平均池化时是否排除光环元素

+ 指定 average_count_excludes_padding=False，均值池化连着填充的 0 一起计算（默认填充的 0 不计入分母）, shape of output tensor 0: (1,1,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            0.25 & 1.25 & 0.75 & 1.  & 1.25 \\
            2.75 & 7.   & 6.   & 6.5 & 7.   \\
            1.25 & 4.   & 3.   & 3.5 & 4.   \\
            1.75 & 4.25 & 3.75 & 4.  & 4.25
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## 三维池化的范例
+ Refer to Pooling3D.py
+ Shape of input tensor 0: (1,1,2,6,9)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             1. &  2. &  3. &  1. &  2. &  3. &  1. &  2. &  3. \\
             4. &  5. &  6. &  4. &  5. &  6. &  4. &  5. &  6. \\
             7. &  8. &  9. &  7. &  8. &  9. &  7. &  8. &  9. \\
             1. &  2. &  3. &  1. &  2. &  3. &  1. &  2. &  3. \\
             4. &  5. &  6. &  4. &  5. &  6. &  4. &  5. &  6. \\
             7. &  8. &  9. &  7. &  8. &  9. &  7. &  8. &  9.
        \end{matrix}\right]
        \left[\begin{matrix}
            10. & 20. & 30. & 10. & 20. & 30. & 10. & 20. & 30. \\
            40. & 50. & 60. & 40. & 50. & 60. & 40. & 50. & 60. \\
            70. & 80. & 90. & 70. & 80. & 90. & 70. & 80. & 90. \\
            10. & 20. & 30. & 10. & 20. & 30. & 10. & 20. & 30. \\
            40. & 50. & 60. & 40. & 50. & 60. & 40. & 50. & 60. \\
            70. & 80. & 90. & 70. & 80. & 90. & 70. & 80. & 90.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (1,1,1,3,4)，最大元素全都来自靠后的通道
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                50. & 60. & 60. & 50. \\
                80. & 90. & 90. & 80. \\
                80. & 90. & 90. & 80.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$
