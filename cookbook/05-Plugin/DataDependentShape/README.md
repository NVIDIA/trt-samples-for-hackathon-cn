# DataDependentShape

+ Move all non-zero elements to the left side.

+ Return 0: the compressed tensor.

+ Return 1: maximum of non-zeros elements among rows.

+ For example:

$$
\text{input} =
\left[\begin{matrix}
    0 & 1 & 2 & 0 \\ 3 & 4 & 5 & 0 \\ 0 & 6 & 0 & 7 \\ 0 & 0 & 0 & 0
\end{matrix}\right]
,
\text{output} =
\left[\begin{matrix}
    1 & 2 & 0 \\ 3 & 4 & 5 \\ 6 & 7 & 0 \\ 0 & 0 & 0
\end{matrix}\right]
$$

+ Steps to run

```shell
make test
```
