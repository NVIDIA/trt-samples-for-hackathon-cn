# paddingNd Layer

+ Steps to run.

```bash
python3 main.py
```

+ IPaddingLayer is deprecated since TensorRT 8.2 and use slice layer instead.

+ Only padding or cropping on the last two dimensions are supported.

+ Ranges of parameters

|         Name         |     Range     |
| :------------------: | :-----------: |
| Rank of input tensor | $\ge 4$ |
