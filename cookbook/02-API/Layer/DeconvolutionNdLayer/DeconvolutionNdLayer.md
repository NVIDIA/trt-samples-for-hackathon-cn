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
+ Do a simple deconvolution on the input tensor.

+ Computation process
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
= {\color{#007F00}{10000.}},
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
={\color{#0000FF}{98765.4321}},
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
={\color{#FF0000}{0.0009}}
$$

+ The computation of the size of input tensor / output tensor / weight / bias is listed [here](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#deconvolution-layer)

+ The value of dimension  C must be buildtime constant in buildtime (rather than -1) even in Dynamic Shape mode.

---

## num_output_maps & kernel_size_nd & kernel & bias

+ Refer to Num_output_Maps+Kernel_size_nd+Kernel+Bias.py
+ Adjust parameters of the deconvolution layer after constructor.

+ Maximum of num_output_maps: 8192

---

## stride_nd

+ Refer to Stride_nd.py
+ Set the moving step of the deconvolution kernel.

+ default value: (1,1)

---

## padding_nd

+ Refer to Padding_nd.py
+ Set the padding element width on deconvolution input.

+ default value: (0,0)

---

## pre_padding

+ Refer to Pre_padding.py
+ Set the head padding element width on deconvolution input.

+ default value: (0,0)

---

## post_padding

+ Refer to Post_padding.py
+ Set the tail padding element width on deconvolution input.

+ default value: (0,0)

---

## padding_mode

+ Refer to Padding_mode.py
+ Set the type of padding element on deconvolution input.

+ Priority of padding_mode is higher than padding_nd.

+ default value: trt.PaddingMode.EXPLICIT_ROUND_DOWN

+ Computation process
[TensorRT C++ API reference](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#a72f43f32e90e4ac5548f8c9ae007584c)

+ Available padding mode
|   trt.PaddingMode   |
| :-----------------: |
|     SAME_UPPER      |
|     SAME_LOWER      |
|  EXPLICIT_ROUND_UP  |
| EXPLICIT_ROUND_DOWN |
|   CAFFE_ROUND_UP    |
|  CAFFE_ROUND_DOWN   |

---

## num_groups

+ Refer to Num_groups.py
+ Set the number of groups on Group-Deconvolution.

+ The numbers of input tensor channel and deconvolution kernel channel should be able to be devided by the number of groups.

+ default value: 1

+ Here we set **num_groupds=2**, devide the dimension C of inpute tensor and deconvoluiton kernel into 2 groups, compute deconvoltion and then concatenate together.

+ In INT8 mode, the number of channel in each group (nC/nGroup and nCOut/nGroup) should be multiple of 4.

---

## 3D Deconvolution

+ Refer to Deconvolution3D.py
+ Deconvolution kernel can move through dimension C.

---

## set_input + INT8-QDQ

+ Refer to Set_input+INT8QDQ.py
+ Use INT8-QDQ mode with set_input API.
