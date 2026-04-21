# RaggedSoftMax Layer

+ Steps to run.

```bash
python3 main.py
```

+ Simple example
## Simple example
+ Refer to Simple.py
+ Shape of input tensor $1: (3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of input tensor $1: (1,3,4,1)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0 \\
        2 \\
        4 \\
        6
    \end{matrix}\right]
    \left[\begin{matrix}
        0 \\
        2 \\
        4 \\
        6
    \end{matrix}\right]
    \left[\begin{matrix}
        0 \\
        2 \\
        4 \\
        6
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (3,4,5). Each batch computes SoftMax with the specified lengths (1,2,3,4), and the remaining elements become 0.
+ When the compute length is 0, the output is all zeros (first row of each batch). When the compute length exceeds the width of input tensor 1, out-of-bounds memory access may occur (red values in the last row of batch 2), so results are undefined.
+ Here it just happens that $0.1862933 = \frac{e^1}{5e^1+e^0}$.
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0.    & 0.    & 0.    & 0.    & 0.    \\
        0.5   & 0.5   & 0.    & 0.    & 0.    \\
        0.25  & 0.25  & 0.25  & 0.25  & 0.    \\
        0.167 & 0.167 & 0.167 & 0.167 & 0.167 \\
    \end{matrix}\right] \\
    \left[\begin{matrix}
        0.    & 0.    & 0.    & 0.    & 0.    \\
        0.5   & 0.5   & 0.    & 0.    & 0.    \\
        0.25  & 0.25  & 0.25  & 0.25  & 0.    \\
        0.167 & 0.167 & 0.167 & 0.167 & 0.167 \\
    \end{matrix}\right] \\
    \left[\begin{matrix}
        0.    & 0.    & 0.    & 0.    & 0.    \\
        0.5   & 0.5   & 0.    & 0.    & 0.    \\
        0.25  & 0.25  & 0.25  & 0.25  & 0.    \\
        {\color{#FF0000}{0.186}} & {\color{#FF0000}{0.186}} & {\color{#FF0000}{0.186}} & {\color{#FF0000}{0.186}} & {\color{#FF0000}{0.186}}
    \end{matrix}\right]
\end{matrix}\right]
$$

+ This layer only accepts 3D tensors for both inputs; otherwise an error is raised:
```
[TRT] [E] 4: [raggedSoftMaxNode.cpp::computeOutputExtents::13] Error Code 4: Internal Error ((Unnamed Layer* 0) [Ragged SoftMax]: Input tensor must have exactly 3 dimensions)
[TRT] [E] 4: (Unnamed Layer* 0) [Ragged SoftMax]: input tensor must have 2 non batch dimensions
[TRT] [E] 4: [network.cpp::validate::2871] Error Code 4: Internal Error (Layer (Unnamed Layer* 0) [Ragged SoftMax] failed validation)
```

+ The two inputs must have the same rank; otherwise an error is raised:
```
[TRT] [E] 3: [network.cpp::addRaggedSoftMax::1294] Error Code 3: API Usage Error (Parameter check failed at: optimizer/api/network.cpp::addRaggedSoftMax::1294, condition: input.getDimensions().nbDims == bounds.getDimensions().nbDims
```
