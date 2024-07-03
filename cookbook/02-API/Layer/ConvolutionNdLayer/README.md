# ConvoutionNd Layer

+ Steps to run.

```bash
python3 main.py
```

+ The count of output channel must be buildtime constant (rather than -1) even in Dynamic Shape mode.
+ Priority of padding APIs: padding_mode > pre_padding = post_padding > padding_nd
+ In group convolution, both the channel count of input tensor and kernel must be able to be divided by the number of groups.
+ In 3D convolution, rank of input tensor must be 5 or more.
+ In INT8 group convolution, the channel count in each group (nC/nGroup and nCOut/nGroup) should be multiple of 4.
+ In 3D Convolution, convolution kernel can move through dimension C.

+ For example:

$$
\left[\quad\begin{matrix}
    \begin{matrix}{\boxed{
        \begin{matrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{matrix}
    }}\end{matrix}\
    \begin{matrix} \;\,1 & \cdots \\ \;\,4 \\ \;\,7 \end{matrix}\\
    \begin{matrix} \ \, 1 & 2 & 3 & 1 & \cdots \\ \ \,\vdots & & & \vdots \end{matrix}
\end{matrix}\right]
\odot
{\boxed{\begin{matrix}
    10^{4} & 10^{3} & 10^{2} \\ 10^{1} & 1 & 10^{-1} \\ 10^{-2} & 10^{-3} & 10^{-4}
\end{matrix}}}
= 12345.6789
\\
\left[\begin{matrix}
    \!\!\!\begin{matrix} 1 \\ 4 \\ 7 \end{matrix}\;\;\
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
= 23156.4897
$$
