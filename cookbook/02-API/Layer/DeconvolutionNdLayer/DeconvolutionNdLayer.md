# DeonvoutionNd Layer

+ Note: **Deconvolution Layer** and related APIs in old TensorRT version are deprecated.

+ Simple example
+ num_output_maps & kernel_size_nd & kernel & bias
+ stride_nd
+ padding_nd
+ pre_padding
+ post_padding
+ padding_mode
+ num_groups
+ 3D Deconvolution
+ set_input + INT8-QDQ

---

## Simple example

+ Refer to SimpleExample.py

+ Shape of input tensor 0: (1,1,3,3)
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

+ By default, shape of output tensor 0: (1,1,5,5).
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

+ Copmputation process
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

+ The computation of the size of input tensor / output tensor / weight / bias is listed [here](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#deconvolution-layer)

+ The value of dimension  C must be constant at buildtime rather than -1.

---

## num_output_maps & kernel_size_nd & kernel & bias

+ Refer to Num_output_Maps+Kernel_size_nd+Kernel+Bias.py, adjust content of the deconvolution layer after constructor.

+ Maximum of num_output_maps: 8192.

+ Shape of output tensor 0: (1,1,5,5), which is the same as default example.

---

## stride_nd

+ Refer to Stride_nd.py, set the moving step of the deconvolution kernel

+ default value: (1,1)

+ Set **stride_nd=(2,2)**, shape of output tensor 0: (1,1,7,7)
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

+ Set **stride_nd=(2,1)**, shape of output tensor 0: (1,1,7,5)
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

+ Set **stride_nd=(1,2)**, shape of output tensor 0: (1,1,5,7)
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

---

## padding_nd

+ Refer to Padding_nd.py, set the padding element width on deconvolution input.

+ default value: (0,0)

+ Set **padding_nd=(1,1)**, shape of output tensor 0: (1,1,3,3)
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

+ Set **padding_nd=(1,0)**, shape of output tensor 0: (1,1,3,5)
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

+ Set **padding_nd=(0,1)**, shape of output tensor 0: (1,1,5,3)
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

+ Set **padding_nd=(2,2)**, shape of output tensor 0: (1,1,1,1)
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

---

## pre_padding

+ Refer to Pre_padding.py, set the head padding element width on deconvolution input.

+ Set **pre_padding=(1,1)**, shape of output tensor 0: (1,1,4,4)
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

+ Set **pre_padding=(1,0)**, shape of output tensor 0: (1,1,4,5)
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

+ Set **pre_padding=(0,1)**, shape of output tensor 0: (1,1,5,4)
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

## post_padding

+ Refer to Post_padding.py , set the tail padding element width on deconvolution input.

+ Set **post_padding=(1,1)**, shape of output tensor 0: (1,1,4,4)
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

+ Set **post_padding=(1,0)**, shape of output tensor 0: (1,1,4,5)
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

+ Set **post_padding=(0,1)**, shape of output tensor 0: (1,1,5,4)
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

## padding_mode

+ Refer to Padding_mode.py, set the type of padding element on deconvolution input.

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

+ Set **padding_mode = trt.PaddingMode.SAME_UPPER**, shape of output tensor 0: (1,1,6,6)
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

+ Set **padding_mode = trt.PaddingMode.SAME_LOWER**, shape of output tensor 0: (1,1,6,6)
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

+ Set **padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP**, shape of output tensor 0: (1,1,7,7)
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

+ Set **padding_mode = trt.PaddingMode.EXPLICIT_ROUND_DOWN**, shape of output tensor 0: (1,1,7,7)
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

+ Set **padding_mode = trt.PaddingMode.CAFFE_ROUND_UP**, shape of output tensor 0: (1,1,7,7)
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

+ Set **padding_mode = trt.PaddingMode.CAFFE_ROUND_DOWN**, shape of output tensor 0: (1,1,7,7)
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

## num_groups

+ Refer to Num_groups.py, the number of groups on Group-Deconvolution.

+ The numbers of input tensor channel and deconvolution kernel channel should be able to be devided by the number of groups.

+ default value: 1

+ Set **num_groupds=2**, devide the dimension C of inpute tensor and deconvoluiton kernel into 2 groups, compute deconvoltion and then concatenate together, shape of output tensor 0: (1,2,4,7)
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

+ In INT8 mode, the number of channel in each group (nC/nGroup and nCOut/nGroup) should be multiple of 4.

---

## 3D Deconvolution

+ Refer to Deconvolution3D.py, deconvolution kernel can move through dimension C.

$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                1.       & 21000.    & 32100.         & 3200.        & 300.     \\
                2.       & 54021.    & 65432.1        & 6503.2       & 600.3    \\
                70040.01 & 87054.02  & 98765.43       & 9806.503     & 900.6003 \\
                   70.04 &    87.054 &    98.76539    &    9.8064995 &   0.9006 \\
                    0.07 &     0.087 &     0.09869999 &    0.0098    &   0.0009
            \end{matrix}\right] \\
            \left[\begin{matrix}
                1. & 0. & 0. & 0. & 0. \\
                2. & 0. & 0. & 0. & 0. \\
                3. & 0.00099945 & 0.0019989 & 0.00007031 & 0. \\
                4. & 0.00000525 & 0.        & 0.00000014 & 0. \\
                5. & 0. & 0. & 0. & 0.
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

## set_input + INT8-QDQ

+ Refer to Set_input+INT8QDQ.py, use INT8-QDQ mode with set_input API.

+ Shape of output tensor 0: (1,1,5,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.   &  381. &  735. &  581. &  300. \\
            2.   & 1164. & 1829. & 1265. &  600. \\
            3.   & 1959. & 2924. & 1949. &  900. \\
              1.   &    0. &    0. &    0. &  0.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$
