# DeonvoutionNd Layer

+ Steps to run.

```bash
python3 main.py
```

+ The number of output channel must be buildtime constant (rather than -1).
+ Priority of padding APIs: padding_mode > pre_padding = post_padding > padding_nd
+ In group deconvolution, both the channel count of input tensor and kernel must be able to be divided by the number of groups.
+ In 3D deconvolution, rank of input tensor must be 5 or more.
+ In INT8 group deconvolution, the channel count in each group (nC/nGroup and nCOut/nGroup) should be multiple of 4.
+ In 3D deconvolution, convolution kernel can move through dimension C.

+ Alternative values of `trt.PaddingMode`, algorithm is [here](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/Deconvolution.html).

|        Name         | Comment |
| :-----------------: | :-----: |
| EXPLICIT_ROUND_DOWN |         |
|  EXPLICIT_ROUND_UP  |         |
|     SAME_LOWER      |         |
|     SAME_UPPER      |         |

+ Ranges of parameters

|         Name         |     Range     |
| :------------------: | :-----------: |
| Rank of input tensor | 4, 5, 6, 7, 8 |
|    Rank of weight    |     4, 5      |
|     Rank of bias     |       1       |

+ Default values of parameters

|     Name     |      Comment       |
| :----------: | :----------------: |
|  padding_nd  |       all 0        |
| pre_padding  | $\left(0,0\right)$ |
| post_padding | $\left(0,0\right)$ |
|  stride_nd   |       all 1        |
| dilation_nd  |       all 1        |

+ Compute process of `case_simple`

$$
\left[\begin{matrix}
    \begin{matrix}{\boxed{
        \begin{matrix} \ & \ & \ \\ \ & \  & \ \\ \ & \ & 1 \end{matrix}
    }}\end{matrix}\
    \begin{matrix} & \\ & \\ \;\, 2 & 3\end{matrix}\\
    \begin{matrix} \ \ \ \ \ \ \, & \  & 4 & 5 & 6 & \\ \ \ \ \ & & 7 & 8 & 9\end{matrix}
\end{matrix}\right]
\odot
{\boxed{\begin{matrix}
    10^{-4} & 10^{-3} & 10^{-2} \\ 10^{-1} & 1 & 10^{1} \\ 10^{2} & 10^{3} & 10^{4}
\end{matrix}}}
= 10000.
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
= 98765.4321
\\
\left[\quad\begin{matrix}
    \begin{matrix}
        1 & 2 & 3 & \ & \ \ \\ 4 & 5 & 6 & \ & \ \
    \end{matrix}\\
    \begin{matrix}
        7 & 8 \\ \ & \ \\ & \
    \end{matrix} \;\
    {\boxed{
        \begin{matrix} \ 9 & \ & \ \\ \ & \ & \ \\ \ & \ & \ \end{matrix}
    }}
\end{matrix}\quad\right]
\odot
{\boxed{\begin{matrix}
    10^{-4} & 10^{-3} & 10^{-2} \\ 10^{-1} & 1 & 10^{1} \\ 10^{2} & 10^{3} & 10^{4}
\end{matrix}}}
= 0.0009
$$
