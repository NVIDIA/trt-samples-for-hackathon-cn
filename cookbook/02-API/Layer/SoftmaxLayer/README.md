# SoftMax Layer

+ Steps to run.

```bash
python3 main.py
```

+ 不能同时指定两个及以上的 axes，如使用 axes=(1<<0)+(1<<1)，会收到报错

+ Default value of axes is `1 << max{0, n-3}`, n is the rank of input tensot.
