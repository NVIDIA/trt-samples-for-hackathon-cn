# CuteDSLPlugin

+ An `IPluginV3` whose kernel is written in **CuteDSL**, CUTLASS's Python DSL.

> **Optional package with a CUDA constraint.** `cupy` ships **its own CUDA runtime**, so the wheel
> has to match your CUDA major version: `cupy-cuda12x` or `cupy-cuda13x`. On this machine
> `cupy 14.2.0` reports CUDA **12.9** while the system toolkit is **13.3** — it works, because it
> uses the libraries it bundles rather than the system ones. That mismatch is tolerable **only**
> because of two properties this cookbook relies on:
>
> 1. `tensorrt_cookbook.utils_plugin` imports `cupy` **lazily, inside the function that needs it**,
>    so `import tensorrt_cookbook` never drags a second CUDA runtime into an unrelated example.
> 2. `tests/run_tests.py` runs **every case in its own subprocess**, so a CuPy build that is wrong
>    for this machine can only break its own case.
>
> Keep both properties when adding examples: import heavy optional packages inside the function,
> never at module scope in a shared module.

+ The operator is RMSNorm, the normalization used by essentially every modern LLM.

+ Requirements: `pip install nvidia-cutlass-dsl` (plus `cupy` and `torch`, already used by
  `05-Plugin/PythonPlugin`) and an **Ampere (SM80) or newer** GPU.

+ Steps to run.

```bash
python3 main.py
```

## Why another Python-plugin backend

`05-Plugin/PythonPlugin` already writes the kernel with NVRTC, CuPy, Numba, Triton and PyTorch.
CuteDSL sits at a different point in that space: the kernel is Python, but it compiles through
CUTLASS and gives direct access to shared memory, warp intrinsics and CuTe layouts — so it can
express the block-level reduction a real LLM kernel needs, which a CuPy or PyTorch elementwise
expression cannot.

The kernel here does exactly that: one block per token, a per-thread partial sum of squares, a tree
reduction in shared memory down to one warp, then `cute.arch.warp_reduction_sum` and
`cute.math.rsqrt`.

## The actual subject: the zero-copy hand-off

TensorRT hands `enqueue()` raw integer device pointers. CuteDSL wants `cute.Tensor` objects. Three
protocol conversions bridge them, none of which copies:

```txt
raw device pointer from TensorRT
  -> cupy.cuda.UnownedMemory   wrap a foreign pointer without owning it  (tensorrt_cookbook.wrap_device_pointer)
  -> torch.as_tensor           via __cuda_array_interface__
  -> cute.runtime.from_dlpack  via __dlpack__, produces the cute.Tensor the kernel sees
```

The first hop is `tensorrt_cookbook.wrap_device_pointer`, shared with `05-Plugin/PythonPlugin`.
"Unowned" matters: the buffers belong to TensorRT, so CuPy must never free them — passing the plugin
as the owner ties the view's lifetime to the plugin instead.

## Launching on TensorRT's stream

A plugin must launch on the stream TensorRT passes into `enqueue()`, not the default stream. CuteDSL
does this by declaring a `CUstream`-typed parameter on the `@cute.jit` launcher; the DSL runtime
picks that argument up as the launch stream, so it is never forwarded to `.launch()` explicitly.
At compile time a `make_fake_stream()` placeholder stands in for it.

## Constexpr vs. runtime values

`hidden_dim` and `epsilon` are `cutlass.Constexpr` and get baked into the binary; `num_tokens` is a
runtime `cutlass.Int32` that only sets the grid dimension. The JIT cache is therefore keyed on
`(hidden_dim, epsilon)` alone, which is what lets **one** compiled kernel serve every sequence
length the optimization profile allows.

`main.py` prints a line on each cache miss, and running two different `num_tokens` through the same
engine shows it firing exactly once:

```txt
--- num_tokens = 128
[CuteDSL] JIT-compiling for key = (1024, 9.999999747378752e-06) (num_tokens = 128 is not part of the key)
[check]:True,...
--- num_tokens = 512
[check]:True,...
```

`clone()` deliberately resets the cache so each execution context owns its own cubins.

## Numerics

The sum of squares is accumulated in FP32 even though `x` and `y` are FP16 — with `hidden_dim` in
the thousands, summing squares in FP16 loses too much precision. `weight` stays FP32 so the scale is
not quantized twice. The result is stored back as FP16, so `check_array` uses a tolerance; the
observed error is about one FP16 ULP (`2^-9` for values in `[1, 2)`), not an algorithmic difference.
