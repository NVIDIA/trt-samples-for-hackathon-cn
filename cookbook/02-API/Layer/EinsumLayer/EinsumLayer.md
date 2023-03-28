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

+ Refer to SimpleExample.py
+ Double tensor single contraction.

+ Computation process
$A_{1\times3\times4}, B_{2\times3\times5}$ do contraction on the dimension of length 3，output is $E_{1\times4\times2\times5}$
$$
\begin{aligned}
C_{1\times4\times3}         &= A^{\text{T}(0,2,1)} \\
D_{1\times2\times4\times5}  &= CB \\
E_{1\times4\times2\times5}  &= D^{\text{T}(0,2,1,3)}
\end{aligned}
$$

---

## equation

+ Refer to Equation.py
+ Adjust the computation expression after constructor.

---

## Transpose

+ Refer to Transpose.py
+ Transpose tensor with Einsum layer.

---

## Sum Reduce

+ Refer to Reduce.py
+ Compute reduce sum on tensor with Einsum layer.

+ Set **equation="ijk->ij"**. The disappeared dimension of right side of "->", k, reduce sum on this dimension, which is equivalent to np.sum(ipnutH0,axis=2).

+ Set **equation="ijk->ik"**, which is equivalent to np.sum(ipnutH0,axis=1)

---

## Dot Product

+ Refer to Dot.py
+ Compute tensor dot product with Einsum layer.

+ Computation process : the dimensions with the same mark on the left side of "->", k, do contraction, and the dimensions disappeared on the right side of "->", i, j, p and q, do reduce sum on these dimension.
$$
0.0 * 1.0 + 0.1 * 1.0 + 2.0 * 1.0 + 0.3 * 1.0 = 6.0
$$

+ substitutive example: change the shape of input tensors as (1,2,4) and (1,3,4). Then the last dimension of each tensor take part in dot product, 2 * 3 =  6 groups in total.
$$
6 + 6 + 6 + 22 + 22 + 22 = 84
$$

+ substitutive example: change both the shape of input tensors and the computation equation. Then similar to the above example, but keep the dimension of j = 2 without reduce sum.

```python
nB0,nH0,nW0 = 1,2,4
nB1,nH1,nW1 = 1,3,4
einsumLayer = network.add_einsum([inputT0, inputT1], "ijk,pqk->j")
```

$$
6 + 6 + 6 = 18，22 + 22 + 22 = 66
$$

---

## Matrix Multiplication

+ Refer to MatrixMultiplication.py
+ Compute matrix multiplication with Einsum layer.

+ Computation process: similar to a batched matric multiplication

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

+ Refer to TripleTensor.py
+ Do contraction on three or more tensors with Einsum layer.

+ Error information:

```txt
[TRT] [E] 3: [layers.cpp::EinsumLayer::5525] Error Code 3: API Usage Error (Parameter check failed at: optimizer/api/layers.cpp::EinsumLayer::5525, condition: nbInputs > 0 && nbInputs <= MAX_EINSUM_NB_INPUTS
```

---

## Take diagonal elements (not support)

+ Refer to Diagonal.py
+ Take diagonal elements deom a tensor with Einsum layer.

+ error informarion:

```txt
[TRT] [E] 3: [layers.cpp::validateEquation::5611] Error Code 3: Internal Error ((Unnamed Layer* 0) [Einsum]: Diagonal operations are not permitted in Einsum equation)
```

---

## Ellipsis (not support)

+ Refer to Ellipsis.py
+ Use ellipsis in computation equation with Einsum layer.

+ Error information:

```txt
[TRT] [E] 3: [layers.cpp::validateEquation::5589] Error Code 3: Internal Error ((Unnamed Layer* 0) [Einsum]: ellipsis is not permitted in Einsum equation)
```
