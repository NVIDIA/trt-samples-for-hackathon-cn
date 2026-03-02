# Pooling Layer

+ Steps to run.

```bash
python3 main.py
```

+ Priority of padding APIs: padding_mode > pre_padding = post_padding > padding_nd

+ Alternative values of `trt.PoolingType`

| name |                 Comment                 |
| :----------------: | :----------------------------------: |
|      AVERAGE       |                             |
|        MAX         |                            |
| MAX_AVERAGE_BLEND  |               |

+ Alternative values of `trt.PaddingMode`, algorithm is [here](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/Convolution.html).

|        Name         | Comment |
| :-----------------: | :-----: |
| EXPLICIT_ROUND_DOWN |         |
|  EXPLICIT_ROUND_UP  |         |
|     SAME_LOWER      |         |
|     SAME_UPPER      |         |

+ Default values of parameters

|     Name     |      Comment       |
| :----------: | :----------------: |
|  padding_nd  |       $\left(0,0\right)$        |
| stride_nd  |  $\left(1,1\right)$ |
| pre_padding |  $\left(0,0\right)$ |
|  post_padding   |        $\left(0,0\right)$        |
| padding_mode  |       trt.PaddingMode.SAME_UPPER        |
| average_count_exclude | False |
| blend_factor | 0.5 |
