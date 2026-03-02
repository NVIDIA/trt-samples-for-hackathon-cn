# Grid Sample Layer

+ Steps to run.

```bash
python3 main.py
```

+ Alternative values of `trt.InterpolationMode`.

|  Name   | Comment |
| :-----: | :-----: |
| NEAREST |         |
| LINEAR  |         |
|  CUBIC  |         |

+ Alternative values of `trt.SampleMode`.

|     Name      | Comment |
| :-----------: | :-----: |
| STRICT_BOUNDS |         |
|     WRAP      |         |
|     CLAMP     |         |
|     FILL      |         |
|    REFLECT    |         |

+ Ranges of parameters

|           Name            |    Range    |
| :-----------------------: | :---------: |
|       align_corners       | True, False |
| Rank of input data tensor |      4      |


+ Default values of parameters

|        Name        |           Comment            |
| :----------------: | :--------------------------: |
|   align_corners    |            False             |
| interpolation_mode | trt.InterpolationMode.LINEAR |
|    sample_mode     |     trt.SampleMode.FILL      |
