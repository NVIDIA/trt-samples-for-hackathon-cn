# ConvoutionNd Layer

+ Note: **Convolution Layer** and related APIs in old TensorRT version are deprecated.

+ Simple example
+ num_output_maps & kernel_size_nd & kernel & bias
+ stride_nd
+ padding_nd
+ pre_padding
+ post_padding
+ padding_mode
+ dilation_nd
+ num_groups
+ 3D Convolution
+ set_input + INT8-QDQ

---

## Simple example

+ Refer to SimpleExample.py

+ Shape of input tensor 0: (1,1,6,9)
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

+ By default, shape of output tensor 0: (1,1,4,7).
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

+ Copmputation process
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

+ The computation of the size of input tensor / output tensor / weight / bias is listed [here](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/Convolution.html)

+ The value of dimension  C must be constant at buildtime rather than -1.

---

## num_output_maps & kernel_size_nd & kernel & bias

+ Refer to Num_output_Maps+Kernel_size_nd+Kernel+Bias.py, adjust content of the convolution layer after constructor.

+ Maximum of num_output_maps: 8192.

+ Shape of output tensor 0: (1,1,4,7), which is the same as default example.

---

## stride_nd

+ Refer to Stride_nd.py, set the moving step of the convolution kernel

+ default value: (1,1)

+ set **stride_nd=(2,2)**, shape of output tensor 0: (1,1,2,4)
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

+ set **stride_nd=(2,1)**, shape of output tensor 0: (1,1,2,7)
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

+ set **stride_nd=(1,2)**, shape of output tensor 0: (1,1,4,4)
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

---

## padding_nd

+ Refer to Padding_nd.py, set the padding element width on convolution input.

+ default value: (0,0)

+ set **padding_nd=(1,1)**, shape of output tensor 0: (1,1,6,9)
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

+ Set **padding_nd=(1,0)**, shape of output tensor 0: (1,1,6,7)
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

+ Set **padding_nd=(0,1)**, shape of output tensor 0: (1,1,4,9)
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

---

## pre_padding

+ Refer to Pre_padding.py, set the head padding element width on convolution input.

+ default value: (0,0)

+ Set **pre_padding=(1,1)**, shape of output tensor 0: (1,1,5,8)
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

+ Set **pre_padding=(1,0)**, shape of output tensor 0: (1,1,5,7)
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

+ Set **pre_padding=(0,1)**, shape of output tensor 0: (1,1,4,8)
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

## post_padding

+ Refer to Post_padding.py, set the tail padding element width on convolution input.

+ default value: (0,0)

+ Set **post_padding=(1,1)**, shape of output tensor 0: (1,1,5,8)
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

+ Set **post_padding=(1,0)**, shape of output tensor 0: (1,1,5,7)
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

+ Set **post_padding=(0,1)**, shape of output tensor 0: (1,1,4,8)
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

## padding_mode

+ Refer to Padding_mode.py, set the type of padding element on convolution input.

+ Priority of padding_mode is higher than padding_nd.

+ default value: trt.PaddingMode.EXPLICIT_ROUND_DOWN

+ The process of computation [TensorRT C++ API reference](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#a72f43f32e90e4ac5548f8c9ae007584c)

+ available padding mode
|   trt.PaddingMode   |
| :-----------------: |
|     SAME_UPPER      |
|     SAME_LOWER      |
|  EXPLICIT_ROUND_UP  |
| EXPLICIT_ROUND_DOWN |
|   CAFFE_ROUND_UP    |
|  CAFFE_ROUND_DOWN   |

+ Set **padding_mode = trt.PaddingMode.SAME_UPPER**, shape of output tensor 0: (1,1,3,5)
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

+ Set **padding_mode = trt.PaddingMode.SAME_LOWER**, shape of output tensor 0: (1,1,3,5)
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

+ Set **padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP**, shape of output tensor 0: (1,1,3,4)
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

+ Set **padding_mode = trt.PaddingMode.EXPLICIT_ROUND_DOWN**, shape of output tensor 0: (1,1,2,4), which is the default padding mode.
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

+ Set **padding_mode = trt.PaddingMode.CAFFE_ROUND_UP**, shape of output tensor 0: (1,1,3,4)
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

+ Set **padding_mode = trt.PaddingMode.CAFFE_ROUND_DOWN**, shape of output tensor 0: (1,1,2,4)
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

## dilation_nd

+ Refer to Dilation_nd.py, Set the adjacent element span of the convolution kernel.

+ default value: (1,1)

+ Set **dilation_nd=(2,2)**, shape of output tensor 0: (1,1,2,5)
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

+ Set **dilation_nd=(2,1)**, shape of output tensor 0: (1,1,2,7)
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

+ Set **dilation_nd=(1,2)**, shape of output tensor 0: (1,1,4,5)
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

---

## num_groups

+ Refer to Num_groups.py, the number of groups on Group-Convolution.

+ The numbers of input tensor channel and convolution kernel channel should be able to be devided by the number of groups.

+ default value: 1

+ Set **num_groupds=2**, devide the dimension C of inpute tensor and convoluiton kernel into 2 groups, compute convoltion and then concatenate together, shape of output tensor 0: (1,2,4,7)
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

+ In INT8 mode, the number of channel in each group (nC/nGroup and nCOut/nGroup) should be multiple of 4.

---

## 3D Convolution

+ Refer to Convolution3D.py, convolution kernel can move through dimension C.

+ Shape of output tensor 0: (1,1,1,4,7), which is equivalent to sum the the two channels of the result of Num_groups.py, then get a result of all zeros.
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

## set_input + INT8-QDQ

+ Refer to Set_input+INT8QDQ.py, use INT8-QDQ mode with set_input API.

+ Shape of output tensor 0: (1,1,4,7)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.   &  791. &  772. &  726. &  791. &  772. &  726. \\
            2.   & 1886. & 1867. & 1821. & 1886. & 1867. & 1821. \\
            3.   & 2882. & 2863. & 2817. & 2882. & 2863. & 2817. \\
            4.   &  791. &  772. &  726. &  791. &  772. &  726.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$
