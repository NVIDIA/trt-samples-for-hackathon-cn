# Elementwise Layer

+ Steps to run.

```bash
python3 main.py
```

+ Case Broadcast works when:
  + The ranks of the two input tensors are same: len(tensor0.shape) == len(tensor1.shape).
  + For each dimension of the two input tensors, either the lengths of this dimension are same, or at least one tensor has length of 1 at this dimension.
