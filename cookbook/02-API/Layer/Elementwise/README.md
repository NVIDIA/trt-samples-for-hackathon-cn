# Elementwise Layer

+ Steps to run.

```bash
python3 main.py
```

+ Alternative values of `trt.ElementWiseOperation`

|   Name    |         Comment         |
| :-------: | :---------------------: |
|    SUM    |                         |
|   PROD    |                         |
|    MAX    |                         |
|    MIN    |                         |
|    SUB    |                         |
|    DIV    |                         |
|    POW    |    can not be int32     |
| FLOOR_DIV |                         |
|    AND    |      must be bool       |
|    OR     |      must be bool       |
|    XOR    |      must be bool       |
|   EQUAL   | can not be bool or int8 |
|  GREATER  | can not be bool or int8 |
|   LESS    | can not be bool or int8 |

+ Case Broadcast works when:
  + The ranks of the two input tensors are same: len(tensor0.shape) == len(tensor1.shape).
  + For each dimension of the two input tensors, either the lengths of this dimension are same, or at least one tensor has length of 1 at this dimension.
