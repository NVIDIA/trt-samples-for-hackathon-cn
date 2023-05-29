# One Hot Layer

+ Common
+ Simple example

---

## Common

+ Input tensor
  + T0, index
  + T1, value
  + T2, depth

+ Output tensor
  + T3, output

+ Data type
  + T0, T1, T2: int32
  + T3: float32, float16

+ Shape
  + T0: [$a_0$, $a_1$, ..., $a_n$]
  + T1: [1]
  + T2: []

+ Attribution and default value
  + axis = -1, axis to insert embedding data.

---

## Simple example

+ Refer to SimpleExample.py

+ Do an embedding on the last dimension of the input tensor.