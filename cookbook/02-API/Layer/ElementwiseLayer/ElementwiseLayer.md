# Elementwise Layer

+ Simple example
+ op
+ Broadcast

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
            1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
\\
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2.
        \end{matrix}\right]
        \left[\begin{matrix}
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2.
        \end{matrix}\right]
        \left[\begin{matrix}
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (1,3,4,5), the two tensor do addition element ny element.
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3.
        \end{matrix}\right]
        \left[\begin{matrix}
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3.
        \end{matrix}\right]
        \left[\begin{matrix}
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3. \\
            3. & 3. & 3. & 3. & 3.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## op

+ Refer to Op.py, adjust content of the assertion layer after constructor.

+ available elementwise operation
| trt.ElementWiseOperation |  $f\left(a,b\right)$   |           Comment            |
| :----------------------: | :--------------------: | :--------------------------: |
|           SUM            |         a + b          |                              |
|           PROD           |         a * b          |                              |
|           MAX            | $\max\left(a,b\right)$ |                              |
|           MIN            | $\min\left(a,b\right)$ |                              |
|           SUB            |         a - b          |                              |
|           DIV            |         a / b          |                              |
|           POW            |   a \*\* b ($a^{b}$)   | Input can not be INT32 type  |
|        FLOOR_DIV         |         a // b         |                              |
|           AND            |        a and b         | Input / Output are Bool type |
|            OR            |         a or b         | Input / Output are Bool type |
|           XOR            |    a ^ b (a xor b)     | Input / Output are Bool type |
|          EQUAL           |         a == b         |     Output is Bool type      |
|         GREATER          |         a > b          |     Output is Bool type      |
|           LESS           |         a < b          |     Output is Bool type      |

+ Do subtraction on the two tensors, shape of output tensor 0: (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0.
        \end{matrix}\right]
        \left[\begin{matrix}
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0.
        \end{matrix}\right]
        \left[\begin{matrix}
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0. \\
            0. & 0. & 0. & 0. & 0.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Error information of using other data type while BOOL type is needed as input:

```txt
[TensorRT] ERROR: 4: [layers.cpp::validate::2304] Error Code 4: Internal Error ((Unnamed Layer* 0) [ElementWise]: operation AND requires boolean inputs.)
```

+ Error information of using non-activation tensor type (datstype beside float32/float16) as input:

```txt
# Both base and exponent are INT32
[TensorRT] ERROR: 4: [layers.cpp::validate::2322] Error Code 4: Internal Error ((Unnamed Layer* 0) [ElementWise]: operation POW requires inputs with activation type.)
# Bse is INT32, exponent is float32
[TensorRT] ERROR: 4: [layers.cpp::validate::2291] Error Code 4: Internal Error ((Unnamed Layer* 0) [ElementWise]: operation POW has incompatible input types Float and Int32)
```

## Broadcast

+ Refer to Broadcast.py, broadcast the element while eementwise operation.

+ Shape of input tensor 0: (1,3,1,5) 和 (1,1,4,1)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
\\
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            2. \\ 2. \\ 2. \\ 2.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (1,3,4,5), do elementwise operation after broadcasting the tensors as the same shape, which is the same as default example.
