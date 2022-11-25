# Fully Connected Layer（deprecated since TensorRT 8.4）

+ Simple example
+ num_output_channels & kernel& bias
+ set_input + INT8-QDQ

---

## Simple example

+ Refer to SimpleExample.py

+ Shape of input tensor 0: (1,3,4,5)
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

+ Shape of output tensor 0: (1,2,1,1)
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

+ Process of computation $output = X \cdot W^{T} + bias$
$$
\begin{aligned}
    &= X.reshape(nB,nC \cdot nH \cdot nW) * W.reshape(nCOut,nC \cdot nH \cdot nW).transpose() \\
    &= \left[\begin{matrix} 0 & 1 & 2 & \cdots & 59 \end{matrix}\right] +
       \left[\begin{matrix} 1 & -1 \\ 1 & -1 \\ \cdots & \cdots \\ 1 & -1 \end{matrix}\right] +
       \left[\begin{matrix} 0 & 0 \end{matrix}\right] \\
    &= \left[\begin{matrix} 1770 & -1770 \end{matrix}\right]
\end{aligned}
$$

+ The length of the last three dimensions of the input tensor must be constant (rather than -1) in buildtime in Dynamic Shape mode.

---

## num_output_channels & kernel  bias

+ Refer to Num_output_channels+Kernel+Bias.py, adjust content of the assertion layer after constructor.

+ Shape of output tensor 0: (1,2,1,1), the same as the default example.

---

+ set_input + INT8-QDQ

+ Refer to Set_input+INT8QDQ.py, use INT8-QDQ mode with set_input API.

+ Shape of output tensor 0: (1,2,1,1), the same as the default example.
