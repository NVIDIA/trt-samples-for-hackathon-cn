# MatrixMultiply Layer

+ Simple example
+ op0 & op1
+ Broadcasting
+ Multiplication of matrix and vector

---

## Simple example

+ Refer to SimpleExample.py
+ Computation process
$$
\left[\begin{matrix}
  0 & 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 & 9 \\ 10 & 11 & 12 & 13 & 14 \\ 15 & 16 & 17 & 18 & 19
\end{matrix}\right]
\left[\begin{matrix}
  1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1
\end{matrix}\right]
=
\left[\begin{matrix}
  {\color{#007F00}{10}} & {\color{#007F00}{10}} & {\color{#007F00}{10}} & {\color{#007F00}{10}} \\ 35 & 35 & 35 & 35 \\ 60 & 60 & 60 & 60 \\ 85 & 85 & 85 & 85
\end{matrix}\right]
$$

---

## op0 & op1

+ Refer to Op0+Op1.py
+ Adjust whether to transpose of the matrix multiplication layer after constructor.

+ Available matrix operation
| trt.MatrixOperation |                        Comment                        |
| :-----------------: | :---------------------------------------------------: |
|        NONE         |        default value, no additional operation         |
|       VECTOR        | note the tensor is a vector (do not use broadcasting) |
|      TRANSPOSE      |   transpose the matrix before matrix multiplication   |

---

## Broadcasting

+ Refer to Broadcast.py
+ $A_{1 \times 3 \times 4 \times 5} \times B_{1 \times 1 \times 4 \times 5} = C_{1 \times 3 \times 4 \times 4}$

---

## Multiplication of matrix and vector

+ Refer to MatrixWithVector.py and MatrixWithVector2.py and result-MatrixWithVector.py
+ Computation process:
$$
\left[\begin{matrix}
  0 & 1 & 2 & 3 & 4\\5 & 6 & 7 & 8 & 9\\10 & 11 & 12 & 13 & 14\\15 & 16 & 17 & 18 & 19
\end{matrix}\right]
\left[\begin{matrix}
  1\\1\\1\\1\\1
\end{matrix}\right]
=
\left[\begin{matrix}
  {\color{#007F00}{10}} \\ {\color{#007F00}{35}} \\ {\color{#007F00}{60}} \\ {\color{#007F00}{85}}
\end{matrix}\right]

\\

\left[\begin{matrix}
  1 & 1 & 1 & 1 & 1
\end{matrix}\right]
\left[\begin{matrix}
  0 & 1 & 2 & 3 & 4\\5 & 6 & 7 & 8 & 9\\10 & 11 & 12 & 13 & 14\\15 & 16 & 17 & 18 & 19
\end{matrix}\right]

=
\left[\begin{matrix}
  {\color{#007F00}{30}} & {\color{#007F00}{34}} & {\color{#007F00}{38}} & {\color{#007F00}{42}} & {\color{#007F00}{46}}
\end{matrix}\right]
$$
