# Identity layer

+ Identity layer.

+ Steps to run.

```bash
python3 main.py
```

+ Copy the input tensor to the output, optionally changing its data type (a cast). Refer to `case_simple` for a plain copy and `case_datatype_conversion` / `case_datatype_conversion_int8` for casting between data types.

+ Input / output data types and shapes:
  + Input tensor `T1` and output tensor `T2` each support [bool, int4, int8, uint8, int32, float8, float16, float32, bfloat16]; `T1` and `T2` may differ (the layer then performs a cast).
  + Input and output have the same shape `[a0, ..., an]`.
