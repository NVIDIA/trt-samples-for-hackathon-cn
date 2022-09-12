# TopK 层
+ 初始范例代码
+ op
+ k
+ axes

---
### 初始范例代码
+ 见 SimpleUSage.py

+ 输入张量形状 (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             9. & 52. &  2. & 27. & 49. \\
             0. & 59. & 22. &  6. & 11. \\
            45. & 33. &  8. & 31. & 37. \\
            23. & 21. &  1. & 55. & 17. \\
        \end{matrix}\right]
        \left[\begin{matrix}
            34. & 15. & 32. & 54. & 39. \\
            10. & 43. & 57. & 30. & 12. \\
            19. & 38. & 40. & 36. & 25. \\
             3. & 42. & 24. & 16. & 47. \\
        \end{matrix}\right]
        \left[\begin{matrix}
            13. & 14. & 58. & 46. & 50. \\
            48. & 44. & 29. & 20. & 18. \\
             4. &  5. & 56. & 28. &  7. \\
            53. & 51. & 41. & 35. & 26. \\
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 0 形状 (1,2,4,5)，对次高维上相同 HW 位置的元素取降序前 2 名
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            34. & 52. & 58. & 54. & 50. \\
            48. & 59. & 57. & 30. & 18. \\
            45. & 38. & 56. & 36. & 37. \\
            53. & 51. & 41. & 55. & 47. \\
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

+ 输出张量 1 形状 (1,2,4,5)，表示输出张量 0 中各元素在输入张量中的通道号
+ $output0 \left[ output1 \left[ c,h,w \right] ,h,w \right] = input \left[c,h,w \right]$
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1 & 0 & 2 & 1 & 2 \\
        2 & 0 & 1 & 1 & 2 \\
        0 & 1 & 2 & 1 & 0 \\
        2 & 2 & 2 & 0 & 1 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        2 & 1 & 1 & 2 & 0 \\
        1 & 2 & 2 & 2 & 1 \\
        1 & 0 & 1 & 0 & 1 \\
        0 & 1 & 1 & 2 & 2 \\
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### op
+ 见 Op.py，在构建 TopK 层后再后再修改其取极值方向

+ 指定 op=trt.TopKOperation.MAX，输出张量 0/1 形状 (1,2,4,5)，结果与初始范例代码相同

+ 指定 op=trt.TopKOperation.MIN，输出张量 0 形状 (1,2,4,5)，对次高维上相同 HW 位置的元素取升序前 2 名
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

+ 指定 op=trt.TopKOperation.MIN，输出张量 1 形状 (1,2,4,5)，表示输出张量 0 中各元素在输入张量中的通道号
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

---
### k
+ 指定 k=2，输出张量 0/1 形状 (1,2,4,5)，结果与初始范例代码相同

+ 最大 k 为 3840，超出后报错
```shell
[TRT] [E] 3: [layers.h::setK::1268] Error Code 3: API Usage Error (Parameter check failed at: /_src/build/cuda-11.4/8.2/x86_64/release/optimizer/api/layers.h::setK::1268, condition: k > 0 && k <= MAX_TOPK_K
)
```

---
### axes
+ 见 Axes.py，在构建 TopK 层后再修改取极值的维度

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
