# Concatenation Layer

+ Simple example
+ axis

---

## Simple example

+ Refer to SimpleExample.py

+ Concatenate on the highest dimension beside Batch dimension.

---

## axis

+ Refer to Axis.py

+ Set the axis of concatenation.

+ default value: min(len(inputT0.shape), 3)

+ set **axis=0** to concatenate on the highest dimension, shape of output tensor 0: (2,3,4,5)

+ set **axis=1** to concatenate on the second highest dimension, shape of output tensor 0: (1,6,4,5), which is the same as default example.

+ set **axis=2** to concatenate on the thrid highest dimension, shape of output tensor 0: (1,3,8,5).

+ set **axis=3** to concatenate on the fourth highest dimension, shape of output tensor 0: (1,3,4,10).
