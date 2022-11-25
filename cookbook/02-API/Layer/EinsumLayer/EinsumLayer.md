# Einsum Layer

+ Simple example
+ equatiuon
+ Transpose
+ Sum Reduce
+ Dot Product
+ Matrix Multiplication
+ Multi-Tensor Contraction (not support)
+ Take diagonal elements (not support)
+ Ellipsis (not support)

---

## Simple example

+ Refer to SimpleExample.py, double tensor single contraction

+ Shape of input tensor 0: (1,3,4) and (2,3,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         1. &  1. &  2. &  3. \\
         2. &  5. &  6. &  7. \\
         3. &  9. & 10. & 11.
    \end{matrix}\right]
\end{matrix}\right]
,
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
         5. &  6. &  7. &  8. &  9. \\
        10. & 11. & 12. & 13. & 14. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        15. & 16. & 17. & 18. & 19. \\
        20. & 21. & 22. & 23. & 24. \\
        25. & 26. & 27. & 28. & 29.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (1,4,2,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
         100. & 112. & 124. & 136. & 148. \\
         280. & 292. & 304. & 316. & 328.
        \end{matrix}\right] \\
        \left[\begin{matrix}
         115. & 130. & 145. & 160. & 175. \\
         340. & 355. & 370. & 385. & 400.
        \end{matrix}\right] \\
        \left[\begin{matrix}
         130. & 148. & 166. & 184. & 202. \\
         400. & 418. & 436. & 454. & 472.
        \end{matrix}\right] \\
        \left[\begin{matrix}
         145. & 166. & 187. & 208. & 229. \\
         460. & 481. & 502. & 523. & 544.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Process of computation：$A_{1\times3\times4}, B_{2\times3\times5}$ do contraction on the dimension of length 3，output is $E_{1\times4\times2\times5}$
$$
\begin{aligned}
C_{1\times4\times3}         &= A^{\text{T}(0,2,1)} \\
D_{1\times2\times4\times5}  &= CB \\
E_{1\times4\times2\times5}  &= D^{\text{T}(0,2,1,3)}
\end{aligned}
$$

---

## equation

+ Refer to Equation.py, adjust content of the assertion layer after constructor.

+ Shape of output tensor 0: (1,4,2,5), which is the same as default example.

---

## Transpose

+ Refer to Transpose.py, transpose ternsor with Einsum layer.

+ Shape of input tensor 0: (1,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
      0. &  1. &  2. &  3. \\
      4. &  5. &  6. &  7. \\
      8. &  9. & 10. & 11. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (3,4,1), which is equivalent to inputH0.transpose(1,2,0)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
      1. \\  1. \\  2. \\  3. \\
    \end{matrix}\right]
    \left[\begin{matrix}
      1. \\  5. \\  6. \\  7. \\
    \end{matrix}\right]
    \left[\begin{matrix}
      1. \\  9. \\ 10. \\ 11. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## Sum Reduce

+ Refer to Reduce.py, reduce sum on tensor with Einsum layer.

+ Shape of input tensor 0: (1,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
      0. &  1. &  2. &  3. \\
      4. &  5. &  6. &  7. \\
      8. &  9. & 10. & 11. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Set **equation="ijk->ij"**, shape of output tensor 0: (1,3) the disappeared dimension of right side of "->", k, reduce sum on this dimension, which is equivalent to np.sum(ipnutH0,axis=2).
$$
\left[\begin{matrix}
     1. & 22. & 38.
\end{matrix}\right]
$$

+ Set **equation="ijk->ik"**, shape of output tensor 0: (1,4), which is equivalent to np.sum(ipnutH0,axis=1)

$$
\left[\begin{matrix}
    1.  & 15. & 18. & 21.
\end{matrix}\right]
$$

---

## Dot Product

+ Refer to Dot.py, tensor dot product with Einsum layer.

+ Shape of input tensor 0: (1,1,4) 和 (1,1,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. &  1. &  2. &  3.
    \end{matrix}\right]
\end{matrix}\right]
,
\left[\begin{matrix}
    \left[\begin{matrix}
        1. &  1. &  1. &  1.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: ()
$$
6.0
$$

+ Process of coputation: the dimensions with the same mark on the left side of "->", k, do contraction, and the dimensions disappeared on the right side of "->", i, j, p and q, do reduce sum on these dimension.

$$
0.0 * 1.0 + 0.1 * 1.0 + 2.0 * 1.0 + 0.3 * 1.0 = 6.0
$$

+ substitutive example (change the shape of input tensors), shape of input tensor 0: (1,2,4) and (1,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. &  1. &  2. &  3. \\
        2. &  5. &  6. &  7.
    \end{matrix}\right]
\end{matrix}\right]
,
\left[\begin{matrix}
    \left[\begin{matrix}
        1. &  1. &  1. &  1. \\
        1. &  1. &  1. &  1. \\
        1. &  1. &  1. &  1.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: ()

$$
84.0
$$

+ Process of computation: the last dimension of each tensor take part in dot product, 2 * 3 =  6 groups in total.
$$
6 + 6 + 6 + 22 + 22 + 22 = 84
$$

```python
# 同时修改输入数据形状和计算表达式
nB0,nH0,nW0 = 1,2,4
nB1,nH1,nW1 = 1,3,4

einsumLayer = network.add_einsum([inputT0, inputT1], "ijk,pqk->j")
```

+ substitutive example (change both the shape of input tensors and the computation equation), shape of output tensor 0: (2,)

$$
\left[\begin{matrix}
    18.  & 66.
\end{matrix}\right]
$$

+ Process of computation: similar to the above example, but keep the dimension of j = 2 without reduce sum.
$$
6 + 6 + 6 = 18，22 + 22 + 22 = 66
$$

---

## Matrix Multiplication

+ Refer to MatrixMultiplication.py, matrix multiplication with Einsum layer.

+ Shape of input tensor 0: (2,2,3) and (2,3,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         1. &  1. &  2. \\
         2. &  4. &  5.
    \end{matrix}\right]
    \left[\begin{matrix}
         1. &  7. &  8. \\
         2. & 10. & 11.
    \end{matrix}\right]
\end{matrix}\right]
,
\left[\begin{matrix}
    \left[\begin{matrix}
         1. &  1. &  1. &  1. \\
         1. &  1. &  1. &  1. \\
         1. &  1. &  1. &  1.
    \end{matrix}\right]
    \left[\begin{matrix}
         1. &  1. &  1. &  1. \\
         1. &  1. &  1. &  1. \\
         1. &  1. &  1. &  1.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (2,2,4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         3. &  3. &  3. &  3. \\
        12. & 12. & 12. & 12.
    \end{matrix}\right]
    \left[\begin{matrix}
        21. & 21. & 21. & 21. \\
        30. & 30. & 30. & 30.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Process of computation: similar to a batched matric multiplication

$$
\left[\begin{matrix}
     1. &  1. &  2. \\
     2. &  4. &  5.
\end{matrix}\right]
\left[\begin{matrix}
     1. &  1. &  1. &  1. \\
     1. &  1. &  1. &  1. \\
     1. &  1. &  1. &  1.
\end{matrix}\right]
=
\left[\begin{matrix}
     3. &  3. &  3. &  3. \\
    12. & 12. & 12. & 12.
\end{matrix}\right]
$$

---

## Multi-Tensor contraction (not support)

+ Refer to TripleTensor.py, do contraction on three or more tensors with Einsum layer.

+ Error information:

```txt
[TRT] [E] 3: [layers.cpp::EinsumLayer::5525] Error Code 3: API Usage Error (Parameter check failed at: optimizer/api/layers.cpp::EinsumLayer::5525, condition: nbInputs > 0 && nbInputs <= MAX_EINSUM_NB_INPUTS
```

---

## Take diagonal elements (not support)

+ Refer to Diagonal.py, take diagonal elements deom a tensor with Einsum layer.

+ error informarion:

```txt
[TRT] [E] 3: [layers.cpp::validateEquation::5611] Error Code 3: Internal Error ((Unnamed Layer* 0) [Einsum]: Diagonal operations are not permitted in Einsum equation)
```

---

## Ellipsis (not support)

+ Refer to Ellipsis.py, use ellipsis in computation equation with Einsum layer.

+ Error information:

```txt
[TRT] [E] 3: [layers.cpp::validateEquation::5589] Error Code 3: Internal Error ((Unnamed Layer* 0) [Einsum]: ellipsis is not permitted in Einsum equation)
```
