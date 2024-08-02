# Fill Layer

+ Steps to run.

```bash
python3 main.py
```

+ Random seed is not supported in TensorRT yet.

+ Alternative values of `trt.FillOperation`.

|      Name      | Comment |
| :------------: | :-----: |
|    LINSPACE    |         |
| RANDOM_NORMAL  |         |
| RANDOM_UNIFORM |         |

+ Default values of parameters

| Name  |      Comment       |
| :---: | :----------------: |
| alpha |         0          |
| beta  |         1          |
