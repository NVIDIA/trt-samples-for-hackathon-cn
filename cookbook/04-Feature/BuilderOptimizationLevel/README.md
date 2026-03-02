# Builder Optimization Level

+ Set optimization level for TRT building.

+ Level 0: Enable a subset of precompiled kernels; disable myelin; select the first tactic sorted by default.
+ Level 1: Enable a subset of precompiled kernels and myelin; test the first 10 tactics selected by a heuristic builder.
+ Level 2: Enable a subset of precompiled kernels and myelin; test all tactics selected.
+ Level 3: **Default configuration**. Enable a larger subset of precompiled kernels and myelin (e.g. Gemm + Reduce); test all tactics seclected.
+ Level 4: Level 3 + Full myelin (offload as many layers as possible to myelin, spending more time).
+ Level 5: Level 4 + Full TRT precompiled kernels (offload as many layers as possible to TRT); compare the TRT and myelin.

+ Steps to run.

```bash
python3 main.py
```
