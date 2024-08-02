# Slice Layer

+ Steps to run.

```bash
python3 main.py
```

+ Alternative values of `trt.SliceMode`
|  Name |                       Comment                        |
| :--------------: | :-----------------------------------------------: |
|     DEFAULT      |             Error if load element out of bound    |
|       WRAP       |                data[i] = data[i%w]                |
|      CLAMP       |                data[i] = data[w-1]                |
|       FILL       |                data[i] = fillValue                |
|     REFLECT      | data[i] = data[(w-1-i%w)*(i/w%2)+(i%w)*(1-i/w%2)] |
