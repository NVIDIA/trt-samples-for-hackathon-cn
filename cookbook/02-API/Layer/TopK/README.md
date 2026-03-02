# TopK Layer

+ Steps to run.

```bash
python3 main.py
```

+ Entry with smaller index will be selected if they own the same value.

+ Axes can not be set more than one, for example, `axes=(1<<2)+(1<<3)`.

+ Alternative values of `trt.TopKOperation`

| Name |         Comment         |
| :--: | :---------------------: |
| MAX  | Get values from maximum |
| MIN  | Get values from minimum |

+ Ranges of parameters

| Name |   Range    |
| :--: | :--------: |
|  K   | $\le 3840$ |
