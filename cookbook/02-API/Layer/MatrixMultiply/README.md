# Matrix Multiply Layer

+ Steps to run.

```bash
python3 main.py
```

+ Alternative values of `trt.MatrixOperation`
| Name |                        Comment                        |
| :-----------------: | :---------------------------------------------------: |
|        NONE         |        no transpose operation         |
|       VECTOR        | the input tensor is vector (do not broadcast) |
|      TRANSPOSE      |   transpose the matrix before multiplication   |
