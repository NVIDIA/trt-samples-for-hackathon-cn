# Elementwise layer

+ Elementwise layer.

+ Steps to run.

```bash
python3 main.py
```

+ Perform an element-by-element operation on two input tensors (with optional broadcast), refer to `case_simple` for the data-type table of each operation and `case_broadcast` for the broadcast rule.

+ Available values of `trt.ElementWiseOperation`.

|   Name    |                             Comment                             |
| :-------: | :-------------------------------------------------------------: |
|    SUM    |                  $output = input1 + input2$                    |
|   PROD    |                $output = input1 \times input2$                 |
|    MAX    |            $output = \max\left(input1, input2\right)$          |
|    MIN    |            $output = \min\left(input1, input2\right)$          |
|    SUB    |                  $output = input1 - input2$                    |
|    DIV    |                  $output = input1 / input2$                    |
|   POWER   |                 $output = input1 ^ {input2}$                   |
| FLOOR_DIV |           $output = \lfloor input1 / input2 \rfloor$           |
|    AND    |     $output = input1\ \mathrm{and}\ input2$ (only for BOOL)    |
|    OR     |     $output = input1\ \mathrm{or}\ input2$ (only for BOOL)     |
|    XOR    |     $output = input1\ \mathrm{xor}\ input2$ (only for BOOL)    |
|   EQUAL   |            $output = \left(input1 = input2\right)$             |
|  GREATER  |           $output = \left(input1 \gt input2\right)$            |
|   LESS    |           $output = \left(input1 \lt input2\right)$            |

+ Data type of input tensors (`T1`) and output tensor (`T2`) depend on the operation:
  + Arithmetic operations (SUM, PROD, MAX, MIN, SUB, DIV, POWER, FLOOR_DIV): `T1` in [int8, int32, int64, float16, float32, bfloat16], `T2` == `T1`.
  + Logical operations (AND, OR, XOR): `T1` == `T2` == bool.
  + Comparison operations (EQUAL, GREATER, LESS): `T1` in [int32, int64, float16, float32, bfloat16], `T2` == bool.

+ Broadcast rule between the two input tensors:
  + The ranks of the two input tensors must be the same: `len(shape0) == len(shape1)`.
  + For each dimension, either the two lengths are equal, or at least one of them is 1.
  + The output length of each dimension is the larger of the two input lengths.
