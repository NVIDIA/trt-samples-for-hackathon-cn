# Parametric ReLU layer

+ Parametric ReLU layer.

+ Steps to run.

```bash
python3 main.py
```

+ A Leaky ReLU whose negative-side slope is defined per element by a second (slopes) input tensor, refer to `case_simple`.

+ Computation process ($slope$ is the corresponding element of the slopes tensor):

$$
output = \begin{cases} input & \left(input \ge 0\right) \\ slope \times input & \left(input \lt 0\right) \end{cases}
$$

+ Input / output data type: T in [int8, float16, float32, bfloat16]; input, slopes and output share dtype T.

+ Shape: input/output shape [a₀,...,aₙ]; slopes shape [b₀,...,bₙ] where each $b_i == a_i$ or $b_i == 1$ (broadcast to the input tensor).
