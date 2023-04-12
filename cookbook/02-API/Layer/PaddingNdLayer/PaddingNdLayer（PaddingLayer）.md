## paddingNd 层（deprecated since TensorRT 8.4）（padding 层 deprecated since TensorRT 8.0）

+ **括号中的层名和参数名适用于 TensorRT8 及之前版本，TensorRT9 及之后被废弃**
+ **PaddingNd 层被标记为废弃，将被 Slice 层取代**
+ Simple example
+ pre_padding_nd (pre_padding) & post_padding_nd (post_padding)
+ 使用 paddingNd 层来进行 crop

---
## Simple example
+ Refer to SimpleExample.py
+ Shape of input tensor 0: (1,3,4,5)
$$
\left[\begin{matrix}
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
            1. & 1. & 1. & 1. & 1.]]]
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (1,3,8,11)，在Input tensor的上、左、下、右分别垫起了 1、2、3、4 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. \\
             0. &  0. &  1. &  2. &  3. &  4. &  5. &  0. &  0. &  0. &  0. \\
             0. &  0. &  6. &  7. &  8. &  9. & 10. &  0. &  0. &  0. &  0. \\
             0. &  0. & 11. & 12. & 13. & 14. & 15. &  0. &  0. &  0. &  0. \\
             0. &  0. & 16. & 17. & 18. & 19. & 20. &  0. &  0. &  0. &  0. \\
             0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. \\
             0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. \\
             0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0.
        \end{matrix}\right]\\
        \left[\begin{matrix}
             0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. \\
             0. &  0. & 21. & 22. & 23. & 24. & 25. &  0. &  0. &  0. &  0. \\
             0. &  0. & 26. & 27. & 28. & 29. & 30. &  0. &  0. &  0. &  0. \\
             0. &  0. & 31. & 32. & 33. & 34. & 35. &  0. &  0. &  0. &  0. \\
             0. &  0. & 36. & 37. & 38. & 39. & 40. &  0. &  0. &  0. &  0. \\
             0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. \\
             0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. \\
             0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0.
        \end{matrix}\right]\\
        \left[\begin{matrix}
             0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. \\
             0. &  0. & 41. & 42. & 43. & 44. & 45. &  0. &  0. &  0. &  0. \\
             0. &  0. & 46. & 47. & 48. & 49. & 50. &  0. &  0. &  0. &  0. \\
             0. &  0. & 51. & 52. & 53. & 54. & 55. &  0. &  0. &  0. &  0. \\
             0. &  0. & 56. & 57. & 58. & 59. & 60. &  0. &  0. &  0. &  0. \\
             0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. \\
             0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. \\
             0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0. &  0.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用旧版 API `add_padding` 会收到警告
```
DeprecationWarning: Use add_padding_nd instead.
```

+ 要求Input tensor维度不小于 3
+ 仅支持 0 元素作为填充元素
+ 仅支持Input tensor的最内两维进行 padding

---

## pre_padding_nd (pre_padding) & post_padding_nd (post_padding)
+ Refer to Pre_padding+Post_padding.py，设置前置和后置的光环元素宽度

+ Shape of output tensor 0: (1,3,8,11)，结果与初始范例代码相同

+ 使用旧版 API `pre_padding` 或 `post_padding` 会收到警告
```
DeprecationWarning: Use pre_padding_nd instead.
DeprecationWarning: Use post_padding_nd instead.
```

+ 目前 padding 只支持 2 维，使用其他维度的 padding 会收到报错
```
[TRT] [E] 3: [network.cpp::addPaddingNd::1243] Error Code 3: API Usage Error (Parameter check failed at: optimizer/api/network.cpp::addPaddingNd::1243, condition: prePadding.nbDims == 2
)
```

---

## 使用 paddingNd 层来进行 crop
+ Refer to Crop.py，使用 Padding 层来实现张量裁剪操作

+ padding 参数可以为负，输出张量尺寸 (1,3,3,3)，Input tensor各 HW 维去掉了首行和末两列

$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             1. &  7. &  8. \\
            1.  & 12. & 13. \\
            2.  & 17. & 18.
        \end{matrix}\right]\\
        \left[\begin{matrix}
            1.  & 27. & 28. \\
            2.  & 32. & 33. \\
            3.  & 37. & 38.
        \end{matrix}\right]\\
        \left[\begin{matrix}
            1.  & 47. & 48. \\
            2.  & 52. & 53. \\
            3.  & 57. & 58.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$