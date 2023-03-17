# Quantize + Dequantize 层
+ Simple example
+ axis
+ set_input 与 zeroPt

---
## Simple example
+ 见SimpleUsage.py

+ Shape of input tensor 0: (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             0. &  1. &  2. &  3. &  4. \\
             5. &  6. &  7. &  8. &  9. \\
            10. & 11. & 12. & 13. & 14. \\
            15. & 16. & 17. & 18. & 19.
        \end{matrix}\right]
        \left[\begin{matrix}
            20. & 21. & 22. & 23. & 24. \\
            25. & 26. & 27. & 28. & 29. \\
            30. & 31. & 32. & 33. & 34. \\
            35. & 36. & 37. & 38. & 39.
        \end{matrix}\right]
        \left[\begin{matrix}
            40. & 41. & 42. & 43. & 44. \\
            45. & 46. & 47. & 48. & 49. \\
            50. & 51. & 52. & 53. & 54. \\
            55. & 56. & 57. & 58. & 59.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              0. &   2. &   4. &   6. &   8. \\
             11. &  13. &  15. &  17. &  19. \\
             21. &  23. &  25. &  28. &  30. \\
             32. &  34. &  36. &  38. &  40.
        \end{matrix}\right]
        \left[\begin{matrix}
             42. &  44. &  47. &  49. &  51. \\
             53. &  55. &  57. &  59. &  61. \\
             63. &  66. &  68. &  70. &  72. \\
             74. &  76. &  78. &  80. &  83.
        \end{matrix}\right]
        \left[\begin{matrix}
             85. &  87. &  89. &  91. &  93. \\
             95. &  97. &  99. & 102. & 104. \\
            106. & 108. & 110. & 112. & 114. \\
            116. & 119. & 121. & 123. & 125.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\begin{aligned}
Quantize: output    &= \textbf{clamp}\left(\textbf{round}\left( \frac{input}{scale}\right ) + zeroPt \right) \\
                    &= \textbf{clamp}\left(\textbf{round}\left( \left[ 0.,1.,2.,...,59. \right] / \frac{60}{127} \right) + 0 \right) \\
                    &= \textbf{clamp}\left( \left[ 0,2,4,...,125 \right] + 0 \right) \\
                    &= \left[ 0,2,4,...,125 \right]
\\
Dequantize: output  &= (input−zeroPt) * scale \\
                    &= \left( \left[ 0,2,4,...,125 \right] - 0\right) * 1. \\
                    &= \left[ 0.,2.,4.,...,125. \right]
\end{aligned}
\\

$$

+ 必须指定量化轴，否则报错：
```
[TensorRT] ERROR: 2: [scaleNode.cpp::getChannelAxis::20] Error Code 2: Internal Error ((Unnamed Layer* 2) [Quantize]: unexpected negative axis)
[TensorRT] ERROR: 2: [scaleNode.cpp::getChannelAxis::20] Error Code 2: Internal Error ((Unnamed Layer* 3) [Dequantize]: unexpected negative axis)
```

---

## axis
+ Refer to Axis.py，指定量化的维度

+ Shape of output tensor 0: (1,3,4,5)，三个通道分别把 [0,60]，[0,120]，[0,240] 映射为 [0,127]（分别大约是除以二、不变、乘以二）
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              0. &   2. &   4. &   6. &   8. \\
             11. &  13. &  15. &  17. &  19. \\
             21. &  23. &  25. &  28. &  30. \\
             32. &  34. &  36. &  38. &  40.
        \end{matrix}\right]
        \left[\begin{matrix}
             21. & 22. & 23. & 24. & 25. \\
             26. & 28. & 29. & 30. & 31. \\
             32. & 33. & 34. & 35. & 36. \\
             37. & 38. & 39. & 40. & 41.
        \end{matrix}\right]
        \left[\begin{matrix}
             21. & 22. & 22. & 23. & 23. \\
             24. & 24. & 25. & 25. & 26. \\
             26. & 27. & 28. & 28. & 29. \\
             29. & 30. & 30. & 31. & 31.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## set_input 与 zeroPt
+ Refer to Set_input+ZeroPt.py，指定量化零点

+ Shape of output tensor 0: (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              0. &   6. &  13. &  19. &  25. \\
             32. &  38. &  44. &  51. &  57. \\
             64. &  70. &  76. &  83. &  89. \\
             95. & 102. & 108. & 114. & 121.
        \end{matrix}\right]
        \left[\begin{matrix}
             64. &  67. &  70. &  73. &  76. \\
             79. &  83. &  86. &  89. &  92. \\
             95. &  98. & 102. & 105. & 108. \\
            111. & 114. & 117. & 121. & 124.
        \end{matrix}\right]
        \left[\begin{matrix}
             85. &  87. &  89. &  91. &  93. \\
             95. &  97. &  99. & 102. & 104. \\
            106. & 108. & 110. & 112. & 114. \\
            116. & 119. & 121. & 123. & 125.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$