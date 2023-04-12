# One Hot Layer

+ Simple example
+ axis

---

## Simple example

+ Refer to SimpleExample.py

+ Input tensor:
  + Index, shape = [$a_0$, $a_1$, ..., $a_n$]
  + Value, shape = [2]
  + Depth, shape = [], one output tensor, Output, and an axis

+ Attribute:
  + axis [scalar]

+ Output tensor
  + output, shape = [$a_0$, $a_1$, ..., $a_{axis-1}$, Depth, $a_{axis}$, $a_{axis+1}$, ..., $a_n$]

---

## axis

+ Refer to Axis.py

+ Adjust content of the one hot layer after constructor.

+ Default value: axis = -1
