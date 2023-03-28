# Select Layer
+ Simple example

---
## Simple example
+ Refer to SimpleExample.py
+ Shape of input tensor $1: (1,3,4,5)，第 0 张量元素全正，第 1 张量的所有元素为第一个的相反数
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

+ Shape of output tensor 0: (1,3,4,5)，交替输出两个输入张量的值
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             -0. &   1. &  -2. &   3. &  -4. \\
              5. &  -6. &   7. &  -8. &   9. \\
            -10. &  11. & -12. &  13. & -14. \\
             15. & -16. &  17. & -18. &  19.
        \end{matrix}\right]
        \left[\begin{matrix}
            -20. &  21. & -22. &  23. & -24. \\
             25. & -26. &  27. & -28. &  29. \\
            -30. &  31. & -32. &  33. & -34. \\
             35. & -36. &  37. & -38. &  39.
        \end{matrix}\right]
        \left[\begin{matrix}
            -40. &  41. & -42. &  43. & -44. \\
             45. & -46. &  47. & -48. &  49. \\
            -50. &  51. & -52. &  53. & -54. \\
             55. & -56. &  57. & -58. &  59.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$