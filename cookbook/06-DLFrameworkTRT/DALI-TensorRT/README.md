# DALI-TensorRT

+ Minimal runnable example of integrating NVIDIA DALI preprocessing with TensorRT inference.

+ The sample pipeline:
  + Generate random image tensor on GPU by DALI.
  + Convert HWC to NCHW and normalize.
  + Feed output into TensorRT identity network.
  + Check output consistency.

+ Steps to run.

```bash
python3 main.py
```
