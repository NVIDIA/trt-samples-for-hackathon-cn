# IfCondition 结构
+ 初始范例代码

---
### 初始范例代码
+ 见 SimpleUsage.py

+ 输入张量 data0 形状 (1,3,4,5)，data1 形状与 data0 相同，其数据为 data0 的相反数
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             1. &  2. &  3. &  4. &  5. \\
             6. &  7. &  8. &  9. & 10. \\
            11. & 12. & 13. & 14. & 15. \\
            16. & 17. & 18. & 19. & 20.
        \end{matrix}\right]
        \left[\begin{matrix}
            21. & 22. & 23. & 24. & 25. \\
            26. & 27. & 28. & 29. & 30. \\
            31. & 32. & 33. & 34. & 35. \\
            36. & 37. & 38. & 39. & 40.
        \end{matrix}\right]
        \left[\begin{matrix}
            41. & 42. & 43. & 44. & 45. \\
            46. & 47. & 48. & 49. & 50. \\
            51. & 52. & 53. & 54. & 55. \\
            56. & 57. & 58. & 59. & 60.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用 data0 作为输入张量时，输出张量形状 (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              2. &   4. &   6. &   8. &  10. \\
             12. &  14. &  16. &  18. &  20. \\
             22. &  24. &  26. &  28. &  30. \\
             32. &  34. &  36. &  38. &  40.
        \end{matrix}\right]
        \left[\begin{matrix}
             42. &  44. &  46. &  48. &  50. \\
             52. &  54. &  56. &  58. &  60. \\
             62. &  64. &  66. &  68. &  70. \\
             72. &  74. &  76. &  78. &  80.
        \end{matrix}\right]
        \left[\begin{matrix}
             82. &  84. &  86. &  88. &  90. \\
             92. &  94. &  96. &  98. & 100. \\
            102. & 104. & 106. & 108. & 110. \\
            112. & 114. & 116. & 118. & 120.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用 data1 作为输入张量时，输出张量形状 (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             1. &  2. &  3. &  4. &  5. \\
             6. &  7. &  8. &  9. & 10. \\
            11. & 12. & 13. & 14. & 15. \\
            16. & 17. & 18. & 19. & 20.
        \end{matrix}\right]
        \left[\begin{matrix}
            21. & 22. & 23. & 24. & 25. \\
            26. & 27. & 28. & 29. & 30. \\
            31. & 32. & 33. & 34. & 35. \\
            36. & 37. & 38. & 39. & 40.
        \end{matrix}\right]
        \left[\begin{matrix}
            41. & 42. & 43. & 44. & 45. \\
            46. & 47. & 48. & 49. & 50. \\
            51. & 52. & 53. & 54. & 55. \\
            56. & 57. & 58. & 59. & 60.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程，等价于以下 python 代码
```python
if inputT0[0,0,0,0] != 0:
    return inputT0 * 2
else:
    return inputT0
```

+ IfCondition 结构的输出来自 IfConditionOutputLayer 层，实际上 IfConditionInputLayer 层和 IfConditionConditionLayer 层也提供了 get_output 方法，但是其输出张量被固定为 None

+ 该结构包含了 ConditionLayer，IfConditionalInputLayer，IfConditionalOutputLayer 层的使用

