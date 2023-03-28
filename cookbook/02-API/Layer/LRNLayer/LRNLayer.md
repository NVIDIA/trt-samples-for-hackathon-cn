# LRN Layer

+ Simple example
+ window_size & alpha & beta & k

---

## Simple example

+ Refer to SimpleExample.py
+ Computation process:
$$
n=3,\alpha=\beta=1.0,k=0.0001
$$

number for sum reduce (use 0 for the element over the edge):$\lfloor \frac{n}{2} \rfloor$

$$
\frac{1^{2}}{ \left( k + \frac{\alpha}{3} \left( 0^{2} + 1^{2} + 2^{2} \right) \right)^{\beta} }
= {\color{#007F00}{0.59996396}},
\\
\frac{2^{2}}{ \left( k + \frac{\alpha}{3} \left( 1^{2} + 2^{2} + 5^{2} \right) \right)^{\beta} }
= {\color{#0000FF}{0.19999799}},
\\
\frac{5^{2}}{ \left( k + \frac{\alpha}{3} \left( 2^{2} + 5^{2} + 0^{2}\right) \right)^{\beta} }
= {\color{#FF0000}{0.51723605}}
$$

---

## window_size & alpha & beta & k

+ Refer to Window_size+Alpha+Bbeta+K.py
+ Adjust type and parameters of the activation layer after constructor.

+ Range of the parameter.

|  parameter  |            Range            |
| :---------: | :-------------------------: |
| window_size | [3,15], must be even number |
|    alpha    |        [-1e20, 1e20]        |
|    beta     |        [0.01, 1e5f]         |
|      k      |        [1e-5, 1e10]         |
