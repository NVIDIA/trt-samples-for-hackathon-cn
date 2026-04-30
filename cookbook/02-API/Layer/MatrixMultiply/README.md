# MatrixMultiply layer

+ MatrixMultiply layer.

+ Steps to run.

```bash
python3 main.py
```

+ Perform batched matrix multiplication $C = A \times B$ on the last two dimensions of the two input tensors, with an independent transform mode (`trt.MatrixOperation`) applied to each operand. Refer to `case_simple` for the basic usage, `case_transpose` for transposing an operand, `case_vector` for treating an operand as a collection of vectors, and `case_broadcast` for broadcasting the batch dimensions.

+ Available values of `trt.MatrixOperation` (applied independently to each operand via `op0` / `op1`).

|   Name    |                                             Comment                                             |
| :-------: | :--------------------------------------------------------------------------------------------: |
|   NONE    |                       Use the operand as-is (a normal $m \times n$ matrix).                       |
| TRANSPOSE |               Transpose the operand; only the last two dimensions are swapped.                |
|  VECTOR   | Treat the operand as a collection of vectors (its last dimension), i.e. one fewer matrix dimension. |

+ Attributes

|  Name  |                Description                | Default |
| :----: | :---------------------------------------: | :-----: |
|  op0   | Transform mode applied to the first operand `A`.  |  NONE   |
|  op1   | Transform mode applied to the second operand `B`. |  NONE   |

+ Input / output data-type and shape constraints:
  + Both input tensors `A` and `B` share the same data type `T` in [float16, float32, bfloat16, int8, float8]. int8 and float8 require explicit quantization.
  + Each input must have rank >= 2. Matrix multiplication is performed on the last two dimensions.
  + For the batch dimensions (all dimensions except the last two), the lengths must match or one of them must be 1 (broadcast).
