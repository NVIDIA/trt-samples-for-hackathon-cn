# Deconvolution layer

+ Deconvolution layer.

+ Steps to run.

```bash
python3 main.py
```

## Attributes

| Attribute | Type | Description | Default | Range / Notes |
|---|---|---|---|---|
| kernel_size | Array | Kernel dimensions per spatial axis | - | 2 or 3 elements; length determines 2D or 3D |
| padding_mode | Enum | Padding calculation method | EXPLICIT_ROUND_DOWN | EXPLICIT_ROUND_DOWN, EXPLICIT_ROUND_UP, SAME_UPPER, SAME_LOWER |
| pre_padding | Array | Pre-padding per spatial dimension | [0, 0, ...] | - |
| post_padding | Array | Post-padding per spatial dimension | [0, 0, ...] | - |
| stride | Array | Stride per spatial dimension | [1, 1, ...] | - |
| dilation | Array | Dilation factor per spatial dimension | [1, 1, ...] | - |
| num_output_maps | int | Number of output feature maps | - | Must be build-time constant |
| num_groups | int | Number of groups in output | 1 | For int8, input and output channel count per group must be multiple of 4 |
| kernel_weights | Weights | Kernel weights | - | Must match input data type |
| bias_weights | Weights | Bias weights | - | Must match input data type |

+ This layer is also known as ConvTranspose, which computes a 2D or 3D deconvolution of an input tensor into an output tensor.

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

+ Compute process of padding mode

  + I = dimensions of input image
  + B = pre-padding, set before output
  + A = post-padding, set after output
  + P = delta between input and output
  + S = stride
  + F = filter
  + O = output
  + D = dilation
  + DK = 1 + D * (F - 1), the data plus any padding

  + EXPLICIT_ROUND_DOWN: Use explicit padding, rounding the output size down.
    $$ O = \left( I - 1 \right) * S + DK - \left( B + A \right) $$
  + EXPLICIT_ROUND_UP: Use explicit padding, rounding the output size up.
    $$ O = \left( I - 1 \right) * S + DK - \left( B + A \right) $$
  + SAME_UPPER: Use SAME padding, with pre-padding ≤ post-padding .
    $$ \begin{aligned}O &= \min\left(I \cdot S, (I - 1) \cdot S + DK\right) \\ P &= \max\left(DK - S, 0\right) \\ B &= \lfloor\frac{P}{2}\rfloor \\ A &= P - B \end{aligned} $$
  + SAME_LOWER: Use SAME padding, with pre-padding ≥ post-padding .
    $$ \begin{aligned}O &= \min\left(I \cdot S, (I - 1) \cdot S + DK\right) \\ P &= \max\left(DK - S, 0\right) \\ A &= \lfloor\frac{P}{2}\rfloor \\ B &= P - A \end{aligned}  $$

+ Compute process of output tensor shape
  + Input tensor: $\left[a_0, a_1, \dots, a_n\right]$.
  + Weight: $\left[k_0, k_1, \dots, k_{m-1}\right],\ 2 \le m \le 3$.
  + Bias: $\left[k\right]$.
  + Output tensor: $\left[b_0, b_1, \dots, b_n\right]$, where:
$$
\begin{aligned}
    s_j &:          \text{stride at spatial dimension j} \\
    k_j &:          \text{kernel at spatial dimension j} \\
    d_j &:          \text{dilation at spatial dimension j} \\
    p_j^{pre} &:    \text{pre padding at spatial dimension j} \\
    p_j^{post} &:   \text{post padding at spatial dimension j} \\
    b_i &=
        \begin{cases}
            a_i &, 0 \le i < n - m \\
            \left( a_i - 1 \right) * s_j + 1 + d_j * \left( k_j - 1 \right) - p_j^{pre} - p_j^{post} &, n - m \le i < n, j = i - \left( n - m \right)
        \end{cases}

\end{aligned}
$$
