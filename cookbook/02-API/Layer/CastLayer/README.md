# Cast Layer

+ Steps to run.

```bash
python3 main.py
```

+ "layer.get_output(0).dtype = XXX" must be set for the data type conversion besides FLOAT32 <-> FLOAT16, or error information below will be received.

```text
Error Code 4: Internal Error (Output tensor (Unnamed Layer* 0) [Cast]_output of type Float produced from output of incompatible type UInt8)
```
