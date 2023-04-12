# Identity Layer

+ Common
+ Simple example
+ use for datatype conversion
+ use for iterator layer

---

## Common

+ Input tensor
  + T0

+ Output tensor
  + T1

+ Data type
  + Combination:
    + float32 | float16 | int32 | bool -> float32 | float16 | int32 | bool
    + float32 | float16 -> uint8
    + uint8 -> float32 | float16

+ Shape
  + T1.shape == T0.shape

+ Attribution and default value

---

## Simple example

+ Refer to SimpleExample.py

---

## use for datatype conversion

+ Refer to DataTypeConversion.py

+ Cast input tensor into FLOAT32 / FLOAT6 / INT32 / INT8.

+ Since TensorRT 8.4, "layer.get_output(0).dtype = XXX" must be set for the data type conversion besides FLOAT32 and FLOAT16, or TensorRT will raize an error.

---

## use for iterator layer

+ Refer to "StaticUnidirectionalLSTM" part of Loop Structure.
