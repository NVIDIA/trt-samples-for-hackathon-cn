# Slice Layer
+ Simple example
+ start & shape & stride
+ mode
+ set_input
    - fill mode + set_input
    - 静态 set_input
    - 动态 set_input
    - dynamic shape 模式下的 slice + set_input
+ 使用 Slice 层取代 Pooling 层

---
## Simple example
+ Refer to SimpleExample.py
+ Shape of input tensor 0: (1,3,4,5)，百位、十位、个位分别表示 CHW 维编号
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              0. &   1. &   2. &   3. &   4. \\
             10. &  11. &  12. &  13. &  14. \\
             20. &  21. &  22. &  23. &  24. \\
             30. &  31. &  32. &  33. &  34.
        \end{matrix}\right]
        \left[\begin{matrix}
            100. & 101. & 102. & 103. & 104. \\
            110. & 111. & 112. & 113. & 114. \\
            120. & 121. & 122. & 123. & 124. \\
            130. & 131. & 132. & 133. & 134.
        \end{matrix}\right]
        \left[\begin{matrix}
            200. & 201. & 202. & 203. & 204. \\
            210. & 211. & 212. & 213. & 214. \\
            220. & 221. & 222. & 223. & 224. \\
            230. & 231. & 232. & 233. & 234.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (1,2,3,4)，以 (0,0,0) 元素为起点，切出 (1,2,3,4) 形状的张量，各维上的步长为 (1,1,1)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              0. &   1. &   2. &   3. \\
             10. &  11. &  12. &  13. \\
             20. &  21. &  22. &  23.
        \end{matrix}\right]
        \left[\begin{matrix}
            100. & 101. & 102. & 103. \\
            110. & 111. & 112. & 113. \\
            120. & 121. & 122. & 123.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## start & shape & stride
+ Refer to Start+Shape+Stride.py，在构建 Slice 层后再后再修改其裁剪起点、形状和步长

+ Shape of output tensor 0: (1,2,3,4)，结果与初始范例代码相同

+ 当 shape 某一维度值为 0 时，该维对应的 start 和 stride 可以任意取值

+ 当被裁减张量某一维度长度为 0 且 shape 该维值非 0 时，会收到报错信息
```
[TRT] [E] 4: (Unnamed Layer* 1) [Slice]: out of bounds slice, input dimensions = [0,1,2,3], start = [0,0,0,0], size = [1,1,2,3], stride = [1,1,1,1].
[TRT] [E] 4: [network.cpp::validate::2917] Error Code 4: Internal Error (Layer (Unnamed Layer* 1) [Slice] failed validation)
```

---

## mode (since TensorRT 7)
+ Refer to Mode.py，修改跨边界元素的裁剪算法

+ 指定 mode=trt.SliceMode.WRAP, shape of output tensor 0: (1,2,3,4)，超出边界的元素从起始元素继续切片
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              0. &   2. &   4. &   1. \\
             20. &  22. &  24. &  21. \\
              0. &   2. &   4. &   1.
        \end{matrix}\right]
        \left[\begin{matrix}
            200. & 202. & 204. & 201. \\
            220. & 222. & 224. & 221. \\
            200. & 202. & 204. & 201.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 mode=trt.SliceMode.CLAMP, shape of output tensor 0: (1,2,3,4)，超出边界的元素取边界值
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              0. &   2. &   4. &   4. \\
             20. &  22. &  24. &  24. \\
             30. &  32. &  34. &  34.
        \end{matrix}\right]
        \left[\begin{matrix}
            200. & 202. & 204. & 204. \\
            220. & 222. & 224. & 224. \\
            230. & 232. & 234. & 234.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 mode=trt.SliceMode.FILL, shape of output tensor 0: (1,2,3,4)，超出边界的元素取固定值，默认值 0
+ 需要设置其他固定值的情况，参见后面的 fill mode + set_input 部分
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              0. &   2. &   4. &   0. \\
             20. &  22. &  24. &   0. \\
              0. &   0. &   0. &   0.
        \end{matrix}\right]
        \left[\begin{matrix}
            200. & 202. & 204. &   0. \\
            220. & 222. & 224. &   0. \\
              0. &   0. &   0. &   0.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 mode=trt.SliceMode.REFLECT, shape of output tensor 0: (1,2,3,4)，超出边界的元素折返取值
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              0. &   2. &   4. &   2. \\
             20. &  22. &  24. &  22. \\
             20. &  22. &  24. &  22.
        \end{matrix}\right]
        \left[\begin{matrix}
            200. & 202. & 204. &   0. \\
            220. & 222. & 224. &   0. \\
            220. & 222. & 224. &   0.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 可用的模式
| trt.SliceMode 名 |                       说明                        |
| :--------------: | :-----------------------------------------------: |
|     DEFAULT      |             默认模式，超出边界就报错              |
|       WRAP       |                data[i] = data[i%w]                |
|      CLAMP       |                data[i] = data[w-1]                |
|       FILL       |                data[i] = fillValue                |
|     REFLECT      | data[i] = data[(w-1-i%w)*(i/w%2)+(i%w)*(1-i/w%2)] |

---

## set_input

### set_input + fill mode
+ Refer to Set_input+ModeFill.py，在填充模式下，指定跨边界元素的填充值

+ Shape of output tensor 0: (1,2,3,4)，超出边界的元素取固定值 -1
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              1. &   2. &   4. &  -1. \\
             1.  &  22. &  24. &  -1. \\
             -1. &  -1. &  -1. &  -1.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.   & 202. & 204. &  -1. \\
            2.   & 222. & 224. &  -1. \\
             -1. &  -1. &  -1. &  -1.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

### 静态 set_input
+ Refer to StaticSlice.py，在构建期以 set_input 的方式向 Slice 层传入切片的参数

+ Shape of output tensor 0: (1,2,3,4)，结果与初始范例代码相同

### 动态 set_input
+ Refer to DynamicSlice.py，在运行期以 set_input 的方式向 Slice 层传入切片的参数

+ Shape of output tensor 0: (1,2,3,4)，结果与静态 set_input 范例代码相同

+ 建立网络时需要 profile，并在运行时绑定真实形状张量的值，否则会有下面几种报错：
```
# 没有用 profile
[TRT] [E] 4: [network.cpp::validate::2919] Error Code 4: Internal Error (Network has dynamic or shape inputs, but no optimization profile has been defined.)
# 没有正确设置 profile
[TRT] [E] 4: [network.cpp::validate::2984] Error Code 4: Internal Error (inputT1: optimization profile is missing values for shape input)
# 没有在运行时绑定形状张量的值
[TRT] [E] 3: [executionContext.cpp::resolveSlots::1481] Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::resolveSlots::1481, condition: allInputShapesSpecified(routine)
)
# 绑定的形状张量的值与 profile 不匹配
[TRT] [E] 3: [executionContext.cpp::setInputShapeBinding::1016] Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::setInputShapeBinding::1016, condition: data[i] <= profileMaxShape[i]. Supplied binding shapes [500,500,500,500] for bindings[3] exceed min ~ max range at index 0, maximum shape in profile is 1, minimum shape in profile is 1, but supplied shape is 500.

)
```

### dynamic shape 模式下的 shuffle + set_input
+ Refer to DynamicShuffleSlice.py，在运行期以 set_input 的方式向 Slice 层传入切片的参数，参数内容可能根据输入张量的变化而变化

+ Shape of output tensor 0: (1,2,3,4)，结果与初始范例代码相同

---

## 使用 Slice 层取代 Pooling 层
+ Refer to Pooling.py，使用 Fill 模式来完成即将被废弃的 PaddingNd 层的功能

+ 输入输出张量形状同“Padding”层的初始范例代码
