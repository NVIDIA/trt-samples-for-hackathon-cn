# Reduce Layer

+ Steps to run.

```bash
python3 main.py
```

+ More than one axis can be set at same time, for example: `axes=(1<<2)+(1<<3)`

+ Alternative values of `trt.ReduceOperation`

| Name | Comment | initialization values (float/int) |
| :--: | :-----: | :-------------------------------: |
| SUM  |   sum   |                 0                 |
| PROD | product |                 1                 |
| AVG  | average |    $-\infty$ / INT_MIN / -128     |
| MAX  | maximum |     $\infty$ / INT_MAX / 127      |
| MIN  | minimum |          NaN / 0 / -128           |
