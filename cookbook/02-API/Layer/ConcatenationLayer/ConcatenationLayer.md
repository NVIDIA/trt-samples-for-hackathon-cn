# Concatenation Layer

+ Common
+ Simple example
+ axis

---

## Common

+ Input tensor
  + list of T0

+ Output tensor
  + T1

+ Data type
  + T0, T1: float32, float16, int32, int8, bool

+ Shape
  + All T0.shape must match excluding the axis dimension.
  + T1.shape is matching the T0.shape excluding the axis dimension, which is the sum of the axis dimension of all T0.

+ Attribution and default value
  + axis = max(len(inputT0.shape) - 3, 0), the axis at which the input tensors should concatenate.
    + In Explicit Batch mode, Axis counts from 0 at the highest dimension of the input tensor (usually the Batch dimension).

    + In Implicit Batch mode, Axis counts from 0 at the highest dimension of the input tensor beside the Batch dimension.

---

## Simple example

+ Refer to SimpleExample.py

+ Concatenate on the highest dimension beside Batch dimension.

---

## axis

+ Refer to Axis.py

+ Set the axis of concatenation.

+ In our example,
  + set **axis=0** to concatenate on the highest dimension, shape of output tensor 0: (2,3,4,5)
  + set **axis=1** to concatenate on the second highest dimension, shape of output tensor 0: (1,6,4,5), which is the same as default example.
  + set **axis=2** to concatenate on the thrid highest dimension, shape of output tensor 0: (1,3,8,5).
  + set **axis=3** to concatenate on the fourth highest dimension, shape of output tensor 0: (1,3,4,10).
