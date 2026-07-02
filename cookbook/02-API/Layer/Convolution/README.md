# Convolution layer

+ Convolution layer.

+ Steps to run.

```bash
python3 main.py
```

+ Attributes

| Attribute | Description | Default |
| --- | --- | --- |
| `num_output_maps` | Number of output maps for the convolution operation. | (required) |
| `kernel_size_nd` | Multi-dimensional kernel size. | (required) |
| `kernel` | Kernel weights. | (required) |
| `bias` | Bias weights. | (optional) |
| `stride_nd` | Multi-dimensional stride. | (1, ..., 1) |
| `dilation_nd` | Multi-dimensional dilation. | (1, ..., 1) |
| `padding_nd` | Multi-dimensional padding. | (0, ..., 0) |
| `pre_padding` | Padding applied before image data. | (0, 0) |
| `post_padding` | Padding applied after image data. | (0, 0) |
| `padding_mode` | Determines how padding is calculated (`EXPLICIT_ROUND_DOWN`, `EXPLICIT_ROUND_UP`, `SAME_UPPER`, `SAME_LOWER`). | `EXPLICIT_ROUND_DOWN` |
| `num_groups` | Number of groups for grouped convolution. | 1 |

+ Compute process of `case_simple`

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

+ Compute process of padding mode

  + I = dimensions of input image
  + B = pre-padding
  + A = post-padding
  + P = delta between input and output
  + S = stride
  + F = filter
  + O = output
  + D = dilation
  + M = I + B + A
  + DK = 1 + D * (F - 1)​​​, the data plus any padding

  + EXPLICIT_ROUND_DOWN: Use explicit padding, rounding the output size down.
    $$ O = \lfloor \frac{M - DK}{S} \rfloor + 1 $$
  + EXPLICIT_ROUND_UP: Use explicit padding, rounding the output size up.
    $$ O = \lceil \frac{M - DK}{S} \rceil + 1 $$
  + SAME_UPPER: Use SAME padding, with pre-padding ≤ post-padding .
    $$ \begin{aligned}O &= \lceil\frac{I}{S}\rceil \\ P &= \lfloor\frac{I-1}{S}\rfloor \cdot S + DK -I \\ B &= \lfloor\frac{P}{2}\rfloor \\ A &= P - B \end{aligned} $$
  + SAME_LOWER: Use SAME padding, with pre-padding ≥ post-padding .
    $$ \begin{aligned}O &= \lceil\frac{I}{S}\rceil \\ P &= \lfloor\frac{I-1}{S}\rfloor \cdot S + DK -I \\ A &= \lfloor\frac{P}{2}\rfloor \\ B &= P - A \end{aligned} $$

+ Compute process of output tensor shape
  + Input tensor: $\left[a_0, a_1, \dots, a_n\right],\ 3 \le n \le 7$.
  + Weight: $\left[k, W_0, \dots, W_m\right],\ 2 \le m \le 3$.
  + Bias: $\left[k\right]$.
  + Output tensor: $\left[a_0, m, b_{n−m}, \dots, b_n\right]$, where:
$$
\begin{aligned}
    m &: \text{num\_output\_maps} \\
    d &: \text{dilation} \\
    k &: \text{kernel\_size} \\
    p &: \text{padding} \\
    b_i&=\lfloor(a_i \cdot p_{i-n+m} - t_{i-n+m})\rfloor + 1 \\
    t_{i-n+m} &= 1 + d_{i-n+m} \cdot (k_{i-n+m} - 1)
\end{aligned}
$$
