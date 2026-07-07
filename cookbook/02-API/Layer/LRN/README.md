# LRN layer

+ LRN layer.

+ Steps to run.

```bash
python3 main.py
```

+ Local Response Normalization (LRN) normalizes each element across a cross-channel window, refer to `case_simple`.

+ Computation process ($w$ is the window size, sum taken over the cross-channel window centered at each position):

$$
output = \frac{input}{\left(k + \alpha \sum input^2\right)^{\beta}}
$$

+ Attributes.

|    Name     |                          Description                          | Default |          Range           |
| :---------: | :----------------------------------------------------------: | :-----: | :----------------------: |
| window_size |         Cross-channel window extent (odd number).           |    -    | {1, 3, 5, 7, 9, 11, 13, 15} (DLA: {3, 5, 7, 9}) |
| alpha       |         Scaling parameter of the normalization divisor.     |    -    | $[-10^{20}, 10^{20}]$    |
| beta        |         Exponent applied to the normalization term.         |    -    | $[0.01, 10^{5}]$         |
| k           |         Stabilization constant to avoid division by 0.      |    -    | $[10^{-5}, 10^{10}]$     |

+ Input / output data type: T in [float32, float16, bfloat16]; input and output share the same shape.

+ Shape: volume $\le 2^{31} - 1$ elements.
