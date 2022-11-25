# MatrixMultiply 层
+ Simple example
+ op0 & op1
+ 乘数广播
+ 矩阵乘向量

---
## Simple example
+ Refer to SimpleExample.py

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

+ Shape of output tensor 0: (1,3,4,4)，各通道上进行矩阵乘法
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         \textcolor[rgb]{0,0.5,0}{10.} & \textcolor[rgb]{0,0.5,0}{10.} & \textcolor[rgb]{0,0.5,0}{10.} & \textcolor[rgb]{0,0.5,0}{10.} \\
         35. & 35. & 35. & 35. \\
         60. & 60. & 60. & 60. \\
         85. & 85. & 85. & 85.
    \end{matrix}\right]
    \left[\begin{matrix}
        110. & 110. & 110. & 110. \\
        135. & 135. & 135. & 135. \\
        160. & 160. & 160. & 160. \\
        185. & 185. & 185. & 185.
    \end{matrix}\right]
    \left[\begin{matrix}
        210. & 210. & 210. & 210. \\
        235. & 235. & 235. & 235. \\
        260. & 260. & 260. & 260. \\
        285. & 285. & 285. & 285.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\left[\begin{matrix}
  0 & 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 & 9 \\ 10 & 11 & 12 & 13 & 14 \\ 15 & 16 & 17 & 18 & 19
\end{matrix}\right]
\left[\begin{matrix}
  1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1
\end{matrix}\right]
=
\left[\begin{matrix}
  \textcolor[rgb]{0,0.5,0}{10} & \textcolor[rgb]{0,0.5,0}{10} & \textcolor[rgb]{0,0.5,0}{10} & \textcolor[rgb]{0,0.5,0}{10} \\ 35 & 35 & 35 & 35 \\ 60 & 60 & 60 & 60 \\ 85 & 85 & 85 & 85
\end{matrix}\right]
$$

---

## op0 & op1
+ Refer to Op0+Op1.py，设定两个乘数是否需要在计算矩阵乘之前做转置

+ Shape of output tensor 0: (1,3,4,4)，结果与初始范例代码相同。第 1 乘数在进行转置操作后再计算矩阵乘法

+ 可用的选项
| trt.MatrixOperation 名 |                          说明                          |
| :--------------------: | :----------------------------------------------------: |
|          NONE          |                   默认行为，不作限制                   |
|         VECTOR         | 指明该参数为向量，不进行元素广播（见“矩阵乘向量”范例） |
|       TRANSPOSE        |               计算乘法前对该矩阵进行转置               |

---

## 乘数广播
+ Refer to Broadcast.py，使用广播特性

+ Shape of output tensor 0: (1,3,4,4)，结果与初始范例代码相同。乘数的形状由 (1,1,5,4) 广播为 (1,3,5,4) 进行计算

---

## 矩阵乘向量
+ Refer to MatrixWithVector.py 和 MatrixWithVector2.py

+ 第一个例子, shape of output tensor 0: (1,3,4)，输入张量右乘向量
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{10.} & \textcolor[rgb]{0,0.5,0}{35.} & \textcolor[rgb]{0,0.5,0}{60.} & \textcolor[rgb]{0,0.5,0}{85.} \\
        110. & 135. & 160. & 185. \\
        210. & 235. & 260. & 285.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\left[\begin{matrix}
  0 & 1 & 2 & 3 & 4\\5 & 6 & 7 & 8 & 9\\10 & 11 & 12 & 13 & 14\\15 & 16 & 17 & 18 & 19
\end{matrix}\right]
\left[\begin{matrix}
  1\\1\\1\\1\\1
\end{matrix}\right]
=
\left[\begin{matrix}
  \textcolor[rgb]{0,0.5,0}{10} \\ \textcolor[rgb]{0,0.5,0}{35} \\ \textcolor[rgb]{0,0.5,0}{60} \\ \textcolor[rgb]{0,0.5,0}{85}
\end{matrix}\right]
$$

+ 第二个例子, shape of output tensor 0: (1,3,5)，输入张量左乘向量
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{30.} & \textcolor[rgb]{0,0.5,0}{34.} & \textcolor[rgb]{0,0.5,0}{38.} & \textcolor[rgb]{0,0.5,0}{42.} & \textcolor[rgb]{0,0.5,0}{46.} \\
        1.   & 114. & 118. & 122. & 126. \\
        2.   & 194. & 198. & 202. & 206.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\left[\begin{matrix}
  1 & 1 & 1 & 1 & 1
\end{matrix}\right]
\left[\begin{matrix}
  0 & 1 & 2 & 3 & 4\\5 & 6 & 7 & 8 & 9\\10 & 11 & 12 & 13 & 14\\15 & 16 & 17 & 18 & 19
\end{matrix}\right]

=
\left[\begin{matrix}
  \textcolor[rgb]{0,0.5,0}{30} & \textcolor[rgb]{0,0.5,0}{34} & \textcolor[rgb]{0,0.5,0}{38} & \textcolor[rgb]{0,0.5,0}{42} & \textcolor[rgb]{0,0.5,0}{46}
\end{matrix}\right]
$$