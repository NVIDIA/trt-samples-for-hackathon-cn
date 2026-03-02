# Data Dependent Shape

+ Example of using a Data-Dependent-Shape plugin to move all non-zero elements to the left side.

+ There are two output tensors from the plugin:
  + 0: the compressed tensor.
  + 1: maximum of non-zeros elements among rows.

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

+ Steps to run.

```bash
make test
```
