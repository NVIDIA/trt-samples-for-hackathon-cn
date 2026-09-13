# PythonPlugin

+ The same as BasicExample, but we make the workflow totally in Python script.

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

+ These examples show 5 ways (using cuda-python, cupy, torch, triton, numba packages respectively) to make it.

+ `add_scalar_multi_tactic.py` goes one step further: it offers Torch **and** Triton as two tactics of a single plugin and lets the builder time them and keep the winner. See the section below.

+ We keep a cuda-python example (add_scalar_cuda_python-V2-deprecated.py) to use deprecated class `IPluginV2DynamicExt`.

+ This example is too simple to show the performance differences among the libraries.

+ Two pieces of boilerplate that every Python plugin needs now live in `tensorrt_cookbook.utils_plugin` instead of being copied into each script:
  + `KernelHelper` / `get_kernel(code, device_id, function_name)` — compile a CUDA source string with **NVRTC** at run time and return a kernel handle (CUBIN for the exact SM when NVRTC supports it, PTX otherwise). Used by the two `add_scalar_cuda_python*.py` examples.
  + `wrap_device_pointer(pointer, shape, dtype, owner)` — view the raw device pointer TensorRT passes into `enqueue()` as a CuPy array **without copying**, via `cupy.cuda.UnownedMemory`. Used by `add_scalar_cupy.py`; the same array hands off zero-copy to PyTorch (`torch.as_tensor`) or CuteDSL (`cute.runtime.from_dlpack`).
  + `check_nvrtc_error(result)` — unwrap the `(status, *values)` tuples that `cuda.bindings.{driver,runtime,nvrtc}` return, raising on a non-zero status.

+ TODO:
  + Remove the redundant memory copy in torch / triton example, which need a solution of wrapping a pointer as a torch.tensor.
  + Get rid of using cupy, so remove the examples with suffix "-using-cupy".
  + Fix numba example, now I get error like below.

```txt
[ERROR] Exception thrown from enqueue() LinkerError: [222] Call to cuLinkAddData results in CUDA_ERROR_UNSUPPORTED_PTX_VERSION
ptxas application ptx input, line 9; fatal   : Unsupported .version 8.4; current version is '8.3'
```

+ Steps to run.

```bash
python3 add_scalar_cuda_python-V2-deprecated.py
python3 add_scalar_cuda_python.py
python3 add_scalar_cupy.py
python3 add_scalar_numba.py
python3 add_scalar_torch.py
python3 add_scalar_triton.py
python3 add_scalar_multi_tactic.py
```

## Letting the builder choose the backend (`add_scalar_multi_tactic.py`)

Every other file here pins one library, so the choice between them is a guess made once,
off-line. `IPluginV3OneBuild.get_valid_tactics` turns it into a measurement: return more than
one tactic and TensorRT times each during `build_serialized_network`, exactly as it does for
its own kernels, then calls `set_tactic` once at run time with the winner.

`get_valid_tactics` is called **after** `configure_plugin`, so it can inspect the I/O type
under consideration and answer differently per format. Here float16 is restricted to Triton,
which expresses "the Torch path is unsuitable for this dtype" without giving up the float32
competition.

Measured on B200, TensorRT 11.1.0.106:

| Format | `get_valid_tactics` returns | `set_tactic` calls while building | baked into the engine |
| ------ | --------------------------- | --------------------------------: | --------------------- |
| float32 | `[Torch, Triton]` | 13 | Torch |
| float16 | `[Triton]` | **0** | Triton |

**A one-element tactic list is not "time one candidate", it is "skip timing entirely".** The
float16 build never calls `set_tactic` at all; the plugin simply runs its single candidate.
That is also why the other files in this directory, which all return `[1]`, never pay any
tuning cost — worth knowing before assuming a plugin is being autotuned.

+ For the same idea in C++, plus a timing cache across builds, see
  [`../Tactic+TimingCache/`](../Tactic+TimingCache/README.md).
