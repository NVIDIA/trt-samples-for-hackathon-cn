# TopK Layer

+ Simple example
+ op
+ k
+ axes
+ K-SetInput (ShapeInputTensor + Data-dependent)

---

## Simple example

+ Refer to SimpleExample.py
+ Output tensor 0: the first two elements with the same position in descending order of the second high dimension.

+ Output tensor 1: the index of the elements of output tensor 0 in the input tensor.

---

## op

+ Refer to Op.py
+ Adjust sort direction of the topk layer after constructor.

+ 指定 op=trt.TopKOperation.MAX，输出张量 0/1 形状 (1,2,4,5)，结果与初始范例代码相同

+ 指定 op=trt.TopKOperation.MIN, shape of output tensor 0: (1,2,4,5)，对次高维上相同 HW 位置的元素取升序前 2 名
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             9. & 14. &  2. & 27. & 39. \\
             0. & 43. & 22. &  6. & 11. \\
             4. &  5. &  8. & 28. &  7. \\
             3. & 21. &  1. & 16. & 17. \\
        \end{matrix}\right]
        \left[\begin{matrix}
            13. & 15. & 32. & 46. & 49. \\
            10. & 44. & 29. & 20. & 12. \\
            19. & 33. & 40. & 31. & 25. \\
            23. & 42. & 24. & 35. & 26. \\
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 op=trt.TopKOperation.MIN, shape of output tensor 1: (1,2,4,5)，表示输出张量 0 中各元素在输入张量中的通道号
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0 & 2 & 0 & 0 & 1 \\
        0 & 1 & 0 & 0 & 0 \\
        2 & 2 & 0 & 2 & 2 \\
        1 & 0 & 0 & 1 & 0 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        2 & 1 & 1 & 2 & 0 \\
        1 & 2 & 2 & 2 & 1 \\
        1 & 0 & 1 & 0 & 1 \\
        0 & 1 & 1 & 2 & 2 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        1 & 0 & 2 & 1 & 2 \\
        2 & 0 & 1 & 1 & 2 \\
        0 & 1 & 2 & 1 & 0 \\
        2 & 2 & 2 & 0 & 1 \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 可用的选项
| trt.TopKOperation 名 |      说明      |
| :------------------: | :------------: |
|         MAX          | 从最大值开始取 |
|         MIN          | 从最小值开始取 |

+ 两种模式下，当输入张量中两个表项具有相同值时，索引较小的表项会优先被选入 TopK 列表中

---

## k
+ 指定 k=2，输出张量 0/1 形状 (1,2,4,5)，结果与初始范例代码相同

+ 最大 k 为 3840，超出后报错
```shell
[TRT] [E] 3: [layers.h::setK::1268] Error Code 3: API Usage Error (Parameter check failed at: /_src/build/cuda-11.4/8.2/x86_64/release/optimizer/api/layers.h::setK::1268, condition: k > 0 && k <= MAX_TOPK_K
)
```

---

## axes
+ Refer to Axes.py，在构建 TopK 层后再修改取极值的维度

+ 指定 axes=1<<0，因为输入张量该维上宽度不足 2，报错：
```
[TRT] [E] 4: (Unnamed Layer* 0) [TopK]: K not consistent with dimensions
[TRT] [E] 4: (Unnamed Layer* 0) [TopK]: K not consistent with dimensions
[TRT] [E] 4: (Unnamed Layer* 0) [TopK]: K not consistent with dimensions
[TRT] [E] 4: [network.cpp::validate::2871] Error Code 4: Internal Error (Layer (Unnamed Layer* 0) [TopK] failed validation)
```

+ 指定 axes=1<<1，输出张量 0/1 形状 (1,2,4,5)，结果与初始范例代码相同

+ 指定 axes=1<<2，输出张量 0/1 形状 (1,3,2,5)，对季高维上相同 CW 位置的元素取降序前 2 名
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            45. & 59. & 22. & 55. & 49. \\
            23. & 52. &  8. & 31. & 37.
        \end{matrix}\right]
        \left[\begin{matrix}
            34. & 43. & 57. & 54. & 47. \\
            19. & 42. & 40. & 36. & 39.
        \end{matrix}\right]
        \left[\begin{matrix}
            53. & 51. & 58. & 46. & 50. \\
            48. & 44. & 56. & 35. & 26.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
\\
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            2 & 1 & 1 & 3 & 0 \\
            3 & 0 & 2 & 2 & 2
        \end{matrix}\right]
        \left[\begin{matrix}
            0 & 1 & 1 & 0 & 3 \\
            2 & 3 & 2 & 2 & 0
        \end{matrix}\right]
        \left[\begin{matrix}
            3 & 3 & 0 & 0 & 0 \\
            1 & 1 & 2 & 3 & 3
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 axes=1<<3，输出张量 0/1 形状 (1,3,4,2)，对叔高维上相同 CH 位置的元素取降序前 2 名
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            52. & 49. \\
            59. & 22. \\
            45. & 37. \\
            55. & 23.
        \end{matrix}\right]
        \left[\begin{matrix}
            54. & 39. \\
            57. & 43. \\
            40. & 38. \\
            47. & 42.
        \end{matrix}\right]
        \left[\begin{matrix}
            58. & 50. \\
            48. & 44. \\
            56. & 28. \\
            53. & 51.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 不能同时指定两个及以上的 axes，如指定 axes=(1<<2)+(1<<3)，会收到报错：
```shell
[TRT] [E] 4: [graphShapeAnalyzer.cpp::computeOutputExtents::1026] Error Code 4: Miscellaneous ((Unnamed Layer* 0) [TopK]: error while computing output extent)
[TRT] [E] 4: [graphShapeAnalyzer.cpp::computeOutputExtents::1026] Error Code 4: Miscellaneous ((Unnamed Layer* 0) [TopK]: error while computing output extent)
[TRT] [E] 3: (Unnamed Layer* 0) [TopK]: reduceAxes must specify exactly one dimension
[TRT] [E] 4: [network.cpp::validate::2871] Error Code 4: Internal Error (Layer (Unnamed Layer* 0) [TopK] failed validation)
```
