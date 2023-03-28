# Shuffle Layer
+ Simple example
+ first_transpose
+ reshape_dims
+ second_transpose
+ 组合使用的例子
+ zero_is_placeholder
+ set_input
    - 静态 set_input
    - 动态 set_input（使用 context.set_shape_input）
    - dynamic shape 模式下的 shuffle + set_input（使用 context.set_input_shape

---
## Simple example
+ Refer to SimpleExample.py
+ Shape of input tensor 0: (1,3,4,5)，百位、十位、个位分别表示 CHW 维编号
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             0. &  1. &  2. &  3. &  4. \\
            10. & 11. & 12. & 13. & 14. \\
            20. & 21. & 22. & 23. & 24. \\
            30. & 31. & 32. & 33. & 34.
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

+ 不指定 shuffle 参数的情况下, shape of output tensor 0: (1,3,4,5)，张量不发生变化
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             0. &  1. &  2. &  3. &  4. \\
            10. & 11. & 12. & 13. & 14. \\
            20. & 21. & 22. & 23. & 24. \\
            30. & 31. & 32. & 33. & 34.
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

---

## first_transpose
+ Refer to First_transpose.py，使用首次转置

+ 指定 first_transpose=(0,2,1，3), shape of output tensor 0: (1,4,3,5)，将第 0、1、2、3 维（原始顺序）分别放置到第 0、2、1、3 维（指定顺序）的位置
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              0. &   1. &   2. &   3. &   4. \\
            100. & 101. & 102. & 103. & 104. \\
            200. & 201. & 202. & 203. & 204.
        \end{matrix}\right]
        \left[\begin{matrix}
             10. &  11. &  12. &  13. &  14. \\
            110. & 111. & 112. & 113. & 114. \\
            210. & 211. & 212. & 213. & 214.
        \end{matrix}\right]
        \left[\begin{matrix}
             20. &  21. &  22. &  23. &  24. \\
            120. & 121. & 122. & 123. & 124. \\
            220. & 221. & 222. & 223. & 224.
        \end{matrix}\right]
        \left[\begin{matrix}
             30. &  31. &  32. &  33. &  34. \\
            130. & 131. & 132. & 133. & 134. \\
            230. & 231. & 232. & 233. & 234.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## reshape_dims
+ Refer to Reshape_dims.py 和 Reshape_dims-IncludeZero.py，使用变形。其中 Reshape_dims.py 使用 -1 自动计算尺寸，Reshape_dims2.py 使用 0 作为 place holder 照搬输入张量该维尺寸

+ 第一个例子，指定 reshape_dims=(-1,2,15), shape of output tensor 0: (2,2,15)，保持原来元素顺序的条件下调整张量形状。可以使用至多一个 -1 自动计算
$$
\left[\begin{matrix}
    \left[\begin{matrix}
          0. & 1. & 2. & 3. & 4. & 10. & 11. & 12. & 13. & 14. & 20. & 21. & 22. & 23. & 24. \\
         30. & 31. & 32. & 33. & 34. & 100. & 101. & 102. & 103. & 104. & 110. & 111. & 112. & 113. & 114.
    \end{matrix}\right]\\
    \left[\begin{matrix}
        120. & 121. & 122. & 123. & 124. & 130. & 131. & 132. & 133. & 134. & 200. & 201. & 202. & 203. & 204. \\
        210. & 211. & 212. & 213. & 214. & 220. & 221. & 222. & 223. & 224. & 230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 第二个例子，指定 reshape_dims=(0,0,-1), shape of output tensor 0: (1,3,20)，0 表示照搬输入张量形状的相应位置上的值，这里两个 0 保持了输入张量形状的最高两维 1 和 3，剩下的自动计算
$$
\left[\begin{matrix}
    \left[\begin{matrix}
          1. &   1. &   2. &   3. &   4. &  10. &  11. &  12. &  13. &  14. &  20. &  21. &  22. &  23. &  24. &  30. &  31. &  32. &  33. &  34.
    \end{matrix}\right]
    \left[\begin{matrix}
        1.   & 101. & 102. & 103. & 104. & 110. & 111. & 112. & 113. & 114. & 120. & 121. & 122. & 123. & 124. & 130. & 131. & 132. & 133. & 134.
    \end{matrix}\right]
    \left[\begin{matrix}
        1.   & 201. & 202. & 203. & 204. & 210. & 211. & 212. & 213. & 214. & 220. & 221. & 222. & 223. & 224. & 230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## second_transpose
+ Refer to Second_transpose.py，使用末次转置

+ 指定 second_transpose=(0,2,1,3), shape of output tensor 0: (1,4,3,5)，单独使用时结果与 first_transpose 范例相同，但是发生在调整形状之后

---

## 组合使用的例子
+ Refer to Combination.py，综合使用首次转置、变形和末次转置

+ Shape of output tensor 0: (1,5,4,3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              1. &   1. &   2. \\
             1.  &  11. &  12. \\
             2.  &  21. &  22. \\
             3.  &  31. &  32.
        \end{matrix}\right]
        \left[\begin{matrix}
              1. &   4. & 100. \\
             1.  &  14. & 110. \\
             2.  &  24. & 120. \\
             3.  &  34. & 130.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.   & 102. & 103. \\
            2.   & 112. & 113. \\
            3.   & 122. & 123. \\
            4.   & 132. & 133.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.   & 200. & 201. \\
            2.   & 210. & 211. \\
            3.   & 220. & 221. \\
            4.   & 230. & 231.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.   & 203. & 204. \\
            2.   & 213. & 214. \\
            3.   & 223. & 224. \\
            4.   & 233. & 234.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              1. &  1. &  2. &  3. &  4. \\
             1.  & 11. & 12. & 13. & 14. \\
             2.  & 21. & 22. & 23. & 24. \\
             3.  & 31. & 32. & 33. & 34.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.   & 101. & 102. & 103. & 104. \\
            2.   & 111. & 112. & 113. & 114. \\
            3.   & 121. & 122. & 123. & 124. \\
            4.   & 131. & 132. & 133. & 134.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.   & 201. & 202. & 203. & 204. \\
            2.   & 211. & 212. & 213. & 214. \\
            3.   & 221. & 222. & 223. & 224. \\
            4.   & 231. & 232. & 233. & 234.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
\\
\Downarrow \ first\_transpose \left( 0,2,1,3 \right)
\\
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              1. &   1. &   2. &   3. &   4. \\
            1.   & 101. & 102. & 103. & 104. \\
            2.   & 201. & 202. & 203. & 204.
        \end{matrix}\right]
        \left[\begin{matrix}
             1.  &  11. &  12. &  13. &  14. \\
            1.   & 111. & 112. & 113. & 114. \\
            2.   & 211. & 212. & 213. & 214.
        \end{matrix}\right]
        \left[\begin{matrix}
             1.  &  21. &  22. &  23. &  24. \\
            1.   & 121. & 122. & 123. & 124. \\
            2.   & 221. & 222. & 223. & 224.
        \end{matrix}\right]
        \left[\begin{matrix}
             1.  &  31. &  32. &  33. &  34. \\
            1.   & 131. & 132. & 133. & 134. \\
            2.   & 231. & 232. & 233. & 234.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
\\
\Downarrow \ reshape \left( 1,4,5,3 \right)
\\
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              1. &   1. &   2. \\
              2. &   4. & 100. \\
            1.   & 102. & 103. \\
            104. & 200. & 201. \\
            105. & 203. & 204.
        \end{matrix}\right]
        \left[\begin{matrix}
             1.  &  11. &  12. \\
             2.  &  14. & 110. \\
            1.   & 112. & 113. \\
            2.   & 210. & 211. \\
            3.   & 213. & 214.
        \end{matrix}\right]
        \left[\begin{matrix}
             1.  &  21. &  22. \\
             2.  &  24. & 120. \\
            1.   & 122. & 123. \\
            2.   & 220. & 221. \\
            3.   & 223. & 224.
        \end{matrix}\right]
        \left[\begin{matrix}
             1.  &  31. &  32. \\
             2.  &  34. & 130. \\
            1.   & 132. & 133. \\
            2.   & 230. & 231. \\
            3.   & 233. & 234.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
\\
\Downarrow \ second\_transpose \left( 0,2,1,3 \right)
\\
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              1. &   1. &   2. \\
             1.  &  11. &  12. \\
             2.  &  21. &  22. \\
             3.  &  31. &  32.
        \end{matrix}\right]
        \left[\begin{matrix}
              1. &   4. & 100. \\
             1.  &  14. & 110. \\
             2.  &  24. & 120. \\
             3.  &  34. & 130.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.   & 102. & 103. \\
            2.   & 112. & 113. \\
            3.   & 122. & 123. \\
            4.   & 132. & 133.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.   & 200. & 201. \\
            2.   & 210. & 211. \\
            3.   & 220. & 221. \\
            4.   & 230. & 231.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.   & 203. & 204. \\
            2.   & 213. & 214. \\
            3.   & 223. & 224. \\
            4.   & 233. & 234.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## zero_is_placeholder
+ Refer to Zero_is_placeholder.py 和 Zero_is_placeholder.py，指定 reshape_dimns 中的 0 值表示 place holder 还是 0 值

+ 第一个例子, shape of output tensor 0: (1,3,4,5)，结果与初始范例代码相同，0 表示照搬输入张量形状的相应位置上的值
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         1. &  1. &  2. &  3. &  4. \\
        1.  & 11. & 12. & 13. & 14. \\
        2.  & 21. & 22. & 23. & 24. \\
        3.  & 31. & 32. & 33. & 34.
    \end{matrix}\right]
    \left[\begin{matrix}
        1.   & 101. & 102. & 103. & 104. \\
        2.   & 111. & 112. & 113. & 114. \\
        3.   & 121. & 122. & 123. & 124. \\
        4.   & 131. & 132. & 133. & 134.
    \end{matrix}\right]
    \left[\begin{matrix}
        1.   & 201. & 202. & 203. & 204. \\
        2.   & 211. & 212. & 213. & 214. \\
        3.   & 221. & 222. & 223. & 224. \\
        4.   & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 第二个例子, shape of output tensor 0: (1,3,4,5)（注意输出张量改成 concatenationLayer.get_output(0)），结果与初始范例代码相同
+ 这种用法常用于本层输出张量广播后再用于其他层的情况，参见 09-Advance 的“EmptyTensor”部分

---

## set_input

### 静态 set_input
+ Refer to StaticShuffle.py，构建期使用 set_input 指定 reshape_dims 的值

+ Shape of output tensor 0: (1,4,5,3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
              1. &   1. &   2. \\
              2. &   4. &  10. \\
             1.  &  12. &  13. \\
             2.  &  20. &  21. \\
             3.  &  23. &  24.
        \end{matrix}\right]
        \left[\begin{matrix}
             1.  &  31. &  32. \\
             2.  &  34. & 100. \\
            1.   & 102. & 103. \\
            2.   & 110. & 111. \\
            3.   & 113. & 114.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.   & 121. & 122. \\
            2.   & 124. & 130. \\
            3.   & 132. & 133. \\
            4.   & 200. & 201. \\
            5.   & 203. & 204.
        \end{matrix}\right]
        \left[\begin{matrix}
            1.   & 211. & 212. \\
            2.   & 214. & 220. \\
            3.   & 222. & 223. \\
            4.   & 230. & 231. \\
            5.   & 233. & 234.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

### 动态 set_input（使用 context.set_shape_input）
+ Refer to DynamicShuffleWithShapeTensor.py，运行期使用 set_input 和 shape tensor 指定 reshape_dims 的值

+ Shape of output tensor 0: (1,4,5,3)，结果与静态 set_input 范例代码相同

+ 建立网络时需要 profile，并在运行时绑定真实形状张量的值，否则会有下面几种报错：
```
# 没有用 profile
[TRT] [E] 4: [network.cpp::validate::2919] Error Code 4: Internal Error (Network has dynamic or shape inputs, but no optimization profile has been defined.)
# 没有正确设置 profile
[TRT] [E] 4: [network.cpp::validate::2984] Error Code 4: Internal Error (inputT1: optimization profile is missing values for shape input)
# 没有在运行时绑定形状张量的值
[TRT] [E] 3: [executionContext.cpp::resolveSlots::1481] Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::resolveSlots::1481, condition: allInputShapesSpecified(routine)
)
# 绑定的形状张量的值与被 shufle 张量形状不匹配
[TRT] [E] 3: [executionContext.cpp::setInputShapeBinding::1016] Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::setInputShapeBinding::1016, condition: data[i] <= profileMaxShape[i]. Supplied binding shapes [2,8,10,6] for bindings[1] exceed min ~ max range at index 1, maximum shape in profile is 5, minimum shape in profile is 1, but supplied shape is 8.

)
```

### dynamic shape 模式下的 shuffle + set_input（使用 context.set_input_shape
+ Refer to DynamicShuffle.py，运行期使用 set_input 指定 reshape_dims 的值

+ 参考输出结果
```
inputH0 : (1, 3, 4, 5)
outputH0: (1, 3, 4, 5, 1)
outputH1: (1, 3, 4, 5)
```