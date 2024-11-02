# Multi-Stream

+ Use one execution context with multiple CUDA stream.

+ Steps to run.

```bash
python3 main.py
```

+ For usage of Page-lock memory, refer to `08-Acvance/StreamAndAsync`.

+ Get timeline to observe

```bash
nsys profile -o Multi-Stream -f true python3 main.py
```
