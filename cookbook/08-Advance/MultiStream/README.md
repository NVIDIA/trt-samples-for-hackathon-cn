# Multi-Stream

+ Use one execution context with multiple CUDA stream.

+ Steps to run.

```bash
python3 main.py
```

+ For usage of Pinned memory, refer to `08-Acvance/PinnedMemory`.

+ Get timeline to observe

```bash
nsys profile -o MultiStream -f true python3 main.py
```
