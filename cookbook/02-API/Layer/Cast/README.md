# Cast Layer

+ Steps to run.

```bash
python3 main.py
```

+ `layer.get_output(0).dtype = XXX` should be used in some conversions or for network input / output tensors.

+ Refer to `../Identity` for more examples of casting data types

+ Alternative values of `trt.DataType`

| Name  |    Alias     |
| :---: | :----------: |
| FLOAT | trt.float32  |
| HALF  | trt.float16  |
| INT8  |   trt.int8   |
| INT32 |   trt.in32   |
| BOOL  |   trt.bool   |
| UINT8 |  trt.uint8   |
|  FP8  |   trt.fp8    |
| BF16  | trt.bfloat16 |
| INT64 |  trt.int64   |
| INT4  |   trt.int4   |
