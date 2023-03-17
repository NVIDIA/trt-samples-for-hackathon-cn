# Identity Layer

+ Simple example
+ use for datatype conversion
+ use for iterator layer

---

## Simple example

+ Refer to SimpleExample.py

---

## use for datatype conversion

+ Refer to DataTypeConversion.py

+ Cast input tensor into FLOAT32 / FLOAT6 / INT32 / INT8.

+ Since TensorRT 8.4, "layer.get_output(0).dtype = XXX" must be set for the data type conversion besides FLOAT32 and FLOAT16, or TensorRT will raize a error.

---

## use for iterator layer

+ Refer to "StaticUnidirectionalLSTM" part of Loop Structure.
