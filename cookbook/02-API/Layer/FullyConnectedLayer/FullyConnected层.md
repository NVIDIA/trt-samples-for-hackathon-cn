# Fully Connected 层（deprecated since TensorRT 8.4）
+ 初始示例代码
+ num_output_channels & kernel& bias
+ set_input + INT8-QDQ 模式

---
### 初始示例代码
+ 见 SimpleUsage.py

+ 输入张量形状 (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 1. & 2. & 3. & 4. \\
            2. & 6. & 7. & 8. & 9. \\
            3.  & 11. & 12. & 13. & 14. \\
            4.  & 16. & 17. & 18. & 19.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.  & 21. & 22. & 23. & 24. \\
            2.  & 26. & 27. & 28. & 29. \\
            3.  & 31. & 32. & 33. & 34. \\
            4.  & 36. & 37. & 38. & 39.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.  & 41. & 42. & 43. & 44. \\
            2.  & 46. & 47. & 48. & 49. \\
            3.  & 51. & 52. & 53. & 54. \\
            4.  & 56. & 57. & 58. & 59.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (1,2,1,1)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                1770.
            \end{matrix}\right]
        \end{matrix}\right] \\
        \left[\begin{matrix}
            \left[\begin{matrix}
                -1770.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：$output = X \cdot W^{T} + bias$
$$
\begin{aligned}
    &= X.reshape(nB,nC \cdot nH \cdot nW) * W.reshape(nCOut,nC \cdot nH \cdot nW).transpose() \\
    &= \left[\begin{matrix} 0 & 1 & 2 & \cdots & 59 \end{matrix}\right] +
       \left[\begin{matrix} 1 & -1 \\ 1 & -1 \\ \cdots & \cdots \\ 1 & -1 \end{matrix}\right] +
       \left[\begin{matrix} 0 & 0 \end{matrix}\right] \\
    &= \left[\begin{matrix} 1770 & -1770 \end{matrix}\right]
\end{aligned}
$$

+ Dynamic Shape 模式下，最低 3 维尺寸必须是构建期常量，不可为 -1

---
### num_output_channels & kernel  bias
+ 见 Num_output_channels+Kernel+Bias.py，在构建 FullyConnected 层后再修改其输出通道数、因子权重、偏置权重

+ 输出张量形状 (1,2,1,1)，结果与初始示例代码相同

---
### set_input + INT8-QDQ 模式
+ 见 Set_input+INT8QDQ.py，使用 set_input 接口和 INT8 QDQ 模式

+ 参考 [link](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_fully_connected_layer.html#aa1eb8deb3192489196cb7884a4177de4)

+ 输出张量形状 (1,2,1,1)，结果与初始示例代码相同

