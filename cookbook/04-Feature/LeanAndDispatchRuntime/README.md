# Lean and Dispatch Rutnime

+ Use Lean and Dispatch Rutnime to do inference.

> **Optional packages.** This example needs `tensorrt-lean` and `tensorrt-dispatch`, which are
> **not** part of the base environment: `pip install tensorrt-lean tensorrt-dispatch`. They are
> separate Python modules (`tensorrt_lean`, `tensorrt_dispatch`) and do not shadow `tensorrt`.
> Without them the affected cases print `[SKIP] tensorrt_lean is unavailable` and the example still
> exits 0.

+ We need packages of `tensorrt_lean` and `tensorrt_dispatch` respectively.

+ Steps to run.

```bash
python3 main.py
```
