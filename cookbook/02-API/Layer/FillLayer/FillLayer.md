# Fill Layer

+ Simple example
+ Linear fill with setting shape in buildtime and range in runtime
+ Linear fill with setting shape and range in runtime
+ Random seed (not support)

---

## Simple example

+ Refer to SimpleExampleLinear.py and SimpleExampleNormalRandom.py and SimpleExampleUniformRandom.py

+ available fill operation
| trt.FillOperation | Input tensor 0 | Input tensor 1 / default value |     Input tensor 2 / default value     |
| :---------------: | :------------: | :----------------------------: | :------------------------------------: |
|     LINSPACE      |  Shape tensor  |  Scalar tensor (Start) / None  |      Scalar tensor (Delta) / None      |
|   RANDOM_NORMAL   |  Shape tensor  |    Scalar tensor (Mean) / 0    | Scalar tensor (standard deviation) / 1 |
|  RANDOM_UNIFORM   |  Shape tensor  |  Scalar tensor (Minimum) / 0   |       Scalar tensor (Maimum) / 1       |

+ LINSPACE mode, shape of output tensor 0: (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1000. & 1001. & 1002. & 1003. & 1004. \\
        1010. & 1011. & 1012. & 1013. & 1014. \\
        1020. & 1001. & 1022. & 1023. & 1024. \\
        1030. & 1031. & 1032. & 1033. & 1034.
    \end{matrix}\right]
    \left[\begin{matrix}
        1100. & 1101. & 1102. & 1103. & 1104. \\
        1110. & 1111. & 1112. & 1113. & 1114. \\
        1120. & 1101. & 1122. & 1123. & 1124. \\
        1130. & 1131. & 1132. & 1133. & 1134.
    \end{matrix}\right]
    \left[\begin{matrix}
        1200. & 1201. & 1202. & 1203. & 1204. \\
        1210. & 1211. & 1212. & 1213. & 1214. \\
        1220. & 1201. & 1222. & 1223. & 1224. \\
        1230. & 1231. & 1232. & 1233. & 1234.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ RANDOM_UNIFORM mode, shape of output tensor 0: (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            -9.167 &  0.371 & -7.195  & 2.565 &  9.814 \\
            -9.129 & -2.469 &  4.144  & 0.099 & -6.012 \\
             9.422 & -9.963 & -2.179  & 8.372 & -9.639 \\
             6.106 &  6.965 & -0.493  & 6.938 & -7.576
        \end{matrix}\right]
        \left[\begin{matrix}
             6.567 &  5.466 & -6.148 & -7.764 & -5.719 \\
             4.527 &  1.752 & -7.469 &  1.24  & -6.424 \\
            -9.2   &  3.142 &  9.268 &  9.176 & -6.118 \\
            -1.818 &  5.001 & -3.764 &  9.836 & -9.384
        \end{matrix}\right]
        \left[\begin{matrix}
            -6.398 &  7.669 & -6.942 & -7.131 &  8.463 \\
            -0.08  & -7.027 &  9.608 &  2.046 & -7.655 \\
             1.096 & -4.69  &  7.327 & -6.187 &  3.415 \\
            -5.887 & -2.402 & -6.263 & -1.868 & -4.79
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ RANDOM_NORMAL mode, shape of output tensor 0: (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            7.0230007 & 3.8968341 & 6.9482727 & 4.1835895 & 5.1206130 \\
            2.7969372 & 6.1688730 & 5.1104345 & 3.8349328 & 4.9340696 \\
            4.7923307 & 3.3197098 & 3.7068114 & 5.1842165 & 7.8175306 \\
            4.8415923 & 4.9943953 & 3.8064980 & 4.4982424 & 3.1755667
        \end{matrix}\right]
        \left[\begin{matrix}
            4.5429380 & 5.6138580 & 3.1937356 & 6.3121758 & 3.8482285 \\
            4.9946094 & 5.0350350 & 7.0324230 & 3.9959793 & 6.6656785 \\
            2.5854793 & 4.4766820 & 5.1439857 & 4.8307447 & 4.3194695 \\
            3.9895120 & 5.0255000 & 4.6139970 & 5.1284995 & 2.8558207
        \end{matrix}\right]
        \left[\begin{matrix}
            6.7800083 & 5.1103050 & 6.4547310 & 6.8914413 & 4.8814700 \\
            3.9176430 & 3.0494800 & 4.9749200 & 5.6528177 & 4.0603640 \\
            5.6039777 & 3.8463345 & 5.4670930 & 3.4261951 & 5.8404710 \\
            4.7249036 & 3.6704760 & 6.2536960 & 6.3368955 & 5.9278727
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Eror information when using linear fill without giving start and delta tensor.

```txt
[TensorRT] ERROR: 2: [fillRunner.cpp::executeLinSpace::46] Error Code 2: Internal Error (Assertion dims.nbDims == 1 failed.Alpha and beta tensor should be set when output an ND tensor)
[TensorRT] INTERNAL ERROR: Assertion failed: dims.nbDims == 1 && "Alpha and beta tensor should be set when output an ND tensor"
```

+ The random seed of random fill is static and solid, meaning the value among multiple times of engine building is the same.

---

## Linear fill with setting shape in buildtime and range in runtime

+ Refer to LinearFill+BuildtimeShape+RuntimeRange.py

+ Shape of output tensor 0: (1,3,4,5), which is the same as default example of linear fill.

---

## Linear fill with setting shape and range in runtime

+ Refer to LinearFill+RuntimeShapeRange.py

+ Shape of output tensor 0: (1,3,4,5), which is the same as default example of linear fill.

---

## Random seed (not support)
