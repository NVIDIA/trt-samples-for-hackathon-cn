# Softmax Layer

+ Steps to run.

```bash
python3 main.py
```

+ Axes can not be set more than one, for example, `axes=(1<<2)+(1<<3)`.

+ Default values of parameters

|     Name     |      Comment       |
| :----------: | :----------------: |
| axes | 1 << $max\left{0, Rank\left(tensor\right)-3\right} |
