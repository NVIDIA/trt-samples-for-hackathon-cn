# TopK Layer

+ Steps to run.

```bash
python3 main.py
```

+ Alternative values of `trt.TopKOperation`

| name |      Comment      |
| :------------------: | :------------: |
|         MAX          | Get values from maximum |
|         MIN          | Get values from minimum|

+ Entry with smaller index will be selected if they own the same value in both modes.

+ Maximum of K is 3840

+ Axes can not be set with more than one, for example, `axes=(1<<2)+(1<<3)`.
