# Cast Layer

+ Common
+ Simple example
+ to_type

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
  + to_type, the data type we want to convert into, can be reset after the constructor

---

## Simple example

+ Refer to SimpleExample.py

+ Cast input tensor into UINT8.

+ "layer.get_output(0).dtype = XXX" must be set for the data type conversion besides FLOAT32 and FLOAT16 Since TensorRT 8.4, or error information below will be received

```text
[TRT] [E] 4: [network.cpp::inferOutputTypes::2048] Error Code 4: Internal Error (Output tensor (Unnamed Layer* 0) [Cast]_output of type Float produced from output of incompatible type UInt8)
```

+ Data type of tensor tensor can be cast among FP32, FP16, INT32, INT8, UINT8, and BOOL.
