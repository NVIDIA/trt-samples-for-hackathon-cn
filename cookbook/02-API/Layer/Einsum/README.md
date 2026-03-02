# Einsum Layer

+ Steps to run.

```bash
python3 main.py
```

+ Case Contraction: $A_{1 \times 3 \times 4} \times B_{2 \times 3 \times 5} \rightarrow C_{1 \times 4 \times 2 \times 5}$

+ Case Transpose: $A_{1 \times 3 \times 4} \rightarrow B_{3 \times 4 \times 1}$

+ Case Sum-Reduce: $A_{1 \times 3 \times 4} \rightarrow B_{1 \times 3}$
  + The disappeared character on the right side of "$\rightarrow$", k, is the axis of sum-reduce operation.
  + More than one axes can be set to do sum-reduce in one layer.

+ Dot Product: $A_{1 \times 1 \times 4} \times B_{1 \times 1 \times 4} \rightarrow c$
  + Computation process: the same character on the left side of "$\rightarrow$", k, is the axis of contraction operation, and the disappeared character on the right side of "$\rightarrow$", i, j, p and q, is the axis of sum-reduce operation.

  $$
  0 * 0 + 1 * 1 + 2 * 2 + 3 * 3 = 6
  $$

  + Substitutive example 1: change the shape of input tensors as (1,2,4) / (1,3,4). So there are 2 * 3 =  6 groups join the sum-reduce operation.

  $$
  6 + 6 + 6 + 22 + 22 + 22 = 84
  $$

  + Substitutive example 2: change the shape of input tensors as (1,2,4) / (1,3,4) and keep "j" in the equation. So the axis "j" was exculed from sum-reduce operation.

  $$
  6 + 6 + 6 = 18，22 + 22 + 22 = 66
  $$
