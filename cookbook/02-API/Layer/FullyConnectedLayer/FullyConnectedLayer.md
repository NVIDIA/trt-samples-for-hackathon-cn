# Fully Connected Layer（deprecated since TensorRT 8.4）

+ Simple example
+ num_output_channels & kernel & bias
+ set_input + INT8-QDQ

---

## Simple example

+ Refer to SimpleExample.py
+ A simple fully connected layer mixing the last three dimensions of the input tensor.

+ Computation process
$$
\begin{aligned}
output &= X \cdot W^{T} + bias \\
    &= X.reshape(nB,nC \cdot nH \cdot nW) * W.reshape(nCout,nC \cdot nH \cdot nW).transpose() \\
    &= \left[\begin{matrix} 0 & 1 & 2 & \cdots & 59 \end{matrix}\right] +
       \left[\begin{matrix} 1 & -1 \\ 1 & -1 \\ \cdots & \cdots \\ 1 & -1 \end{matrix}\right] +
       \left[\begin{matrix} 0 & 0 \end{matrix}\right] \\
    &= \left[\begin{matrix} 1770 & -1770 \end{matrix}\right]
\end{aligned}
$$

+ The length of the last three dimensions of the input tensor must be buildtime constant in buildtime (rather than -1) even in Dynamic Shape mode.

---

## num_output_channels & kernel & bias

+ Refer to Num_output_channels+Kernel+Bias.py
+ Adjust the number of output channels, weight and bias of the FC layer after constructor.

---

## set_input + INT8-QDQ

+ Refer to Set_input+INT8QDQ.py
+ Use INT8-QDQ mode with set_input API.
