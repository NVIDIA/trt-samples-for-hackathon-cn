# Multi-Stream

+ An example of using one context with multi CUDA stream.

+ For usage of Page-lock memory, refer to `08-Acvance/StreamAndAsync`.

+ Steps to run

```bash
python3 main.py
```

+ Get timeline to observe

```bash
nsys profile -o Multi-Stream -f true python3 main.py
```
