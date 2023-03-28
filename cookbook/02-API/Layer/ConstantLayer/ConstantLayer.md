# Activation Layer

+ Simple example
+ weights & shape

---

## Common

+ Input tensor
  + None

+ Output tensor
  + T0

+ Data type
  + T0: float32, float16, int32, int8 (in QDQ mode), bool (since TensorRT 8.6)

+ Shape
  + T0: set by parameter shape

+ Attribution and default value
  + shape: shape of the output tensor.

  + weights: data content of the output tensor.
    + Volume(weights) == np.prod(shape) must be hold
    + np.ascontiguousarray() must be used while converting np.array to trt.Weights, or the output of the constant layer could be incorrect.

---

## Simple example

+ Refer to SimpleExample.py

+ Produce a constant tensor during build time.

---

## weights & shape

+ Refer to Weight+Shape.py

+ Adjust data and shape of the constant tensor after constructor.

## Data Type Int8

+ Refer to DataTypeInt8.py

+ Use a int8 constant tensor in Int8-QDQ mode.

+ BuilderFlag.INT8 must be used for Int8 mode, or error information below will be received.

```txt
[TRT] [E] 4: [network.cpp::validate::2925] Error Code 4: Internal Error (Int8 precision has been set for a layer or layer output, but int8 is not configured in the builder)
txt

+ BuilderFlag.OBEY_PRECISION_CONSTRAINTS must be used, or error information below will be received.

```txt
[TRT] [E] 4: [qdqGraphOptimizer.cpp::matchInt8ConstantDQ::3721] Error Code 4: Internal Error (: Int8 constant usage is restricted to explicit precision mode)
```

+ Dequantize layer should be used, or error information below will be received.

```txt
[TRT] [E] 4: [qdqGraphOptimizer.cpp::matchInt8ConstantDQ::3725] Error Code 4: Internal Error (: Int8 constant is only allowed before DQ node)
```

## Data Type Bool

+ Refer to DataTypeBool.py

+ Use a bool constant layer.

+ this feature is supported since TensorRT 8.6.

+ Output data type must be set as trt.bool, or error information below will be received.

```txt
[TRT] [E] 4: [network.cpp::inferOutputTypes::2048] Error Code 4: Internal Error (Output tensor (Unnamed Layer* 0) [Constant]_output of type Float produced from output of incompatible type Bool)
```
