# Cast Layer

+ "layer.get_output(0).dtype = XXX" must be set for the data type conversion besides FLOAT32 <-> FLOAT16, or error information below will be received.

```text
Error Code 4: Internal Error (Output tensor (Unnamed Layer* 0) [Cast]_output of type Float produced from output of incompatible type UInt8)
```

## Case Simple

+ A simple case to cast float32 tensor into uint 8 tensor.
