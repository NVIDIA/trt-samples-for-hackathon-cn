# DebugTensor - C++

+ The C++ side of debug tensors, using two helpers from `include/cookbookHelper.cuh`.

+ Steps to run.

```bash
make all
./main.exe
```

## `DebugTensorWriter`

An `IDebugListener` that writes every reported debug tensor to a **`.npy` file** (through the
bundled `include/cnpy.*`) instead of only printing it, so the intermediate values can be diffed
against a reference run in numpy afterwards:

```txt
[DebugTensorWriter] a_cute_tensor, FP32 , (3, 4, 5), 240 bytes -> debug-tensor-a_cute_tensor.npy
```

```python
>>> import numpy as np
>>> np.load("debug-tensor-a_cute_tensor.npy")
array([[[ 0.,  2.,  4.,  6.,  8.], ...
```

The Python counterpart, `CookbookDebugListener` (used by `04-Feature/DebugTensor/main.py`), prints
the tensor and optionally compares it against an expected value; this one persists it.

Types numpy has no direct equivalent for (FP16 / BF16 / FP8 / INT4 …) are written as a flat `uint8`
array of the raw bytes, so they can still be reinterpreted on the numpy side.

## `launchGlobalTimerKernel`

A one-thread kernel that reads the PTX **`%globaltimer`** register — the GPU-wide nanosecond clock —
and stores it to device memory. Recording it before and after the work being measured gives an
elapsed time that does not depend on `cudaEventElapsedTime()`, which is **documented as unreliable
when Confidential Compute is enabled**.

The example records both so they can be compared:

```txt
cudaEventElapsedTime : 3.86237 ms
%globaltimer         : 3.79373 ms
```

They agree closely here (Confidential Compute is off). The point is that the second measurement
remains trustworthy in an environment where the first one does not.

## Note on `markDebug`

`network->markDebug(tensor)` keeps the marked tensor from being fused away, which changes the
optimized graph and can cost performance. `markUnfusedTensorsAsDebugTensors` is the alternative that
leaves fusion decisions alone — see the discussion at the end of `04-Feature/DebugTensor/main.py`.
