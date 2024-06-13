# Identity Layer

## Case Simple

+ A simple case of using Identity layer.

## Case Data type conversion

+ Cast input tensor into FLOAT32 / FLOAT16 / INT32 / INT64 / INT8 / UINT8 / INT4 / BOOL
+ Since TensorRT-8.4, "layer.get_output(0).dtype = XXX" must be set for the data type conversion besides FLOAT32 and FLOAT16.
