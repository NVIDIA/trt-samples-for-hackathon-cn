# nvtripy

+ An eager-mode Python frontend for TensorRT, installed into its own virtual environment.

**Tripy** (package `nvtripy`, from [NVIDIA/TensorRT-Incubator](https://github.com/NVIDIA/TensorRT-Incubator))
lets you write the model as a `tp.Module`, run it immediately while debugging, then `tp.compile`
it into a TensorRT `Executable`.

+ GitHub [Link](https://github.com/NVIDIA/TensorRT-Incubator/tree/main/tripy) ·
  [Document](https://nvidia.github.io/TensorRT-Incubator/pre0_user_guides/00-introduction-to-tripy.html)
  (the package name is `nvtripy`, not `tripy`).

+ Steps to run.

```bash
python3 main.py
```

Measured with nvtripy **0.1.7** on B200. Tripy is **pre-1.0 and its API is still moving** —
treat this as an "experimental frontend" tour, not a stable interface.

## Read this before you `pip install nvtripy`

**Installing nvtripy into the cookbook's environment breaks the cookbook.** It is not a pure-Python
package: it depends on `tensorrt-cu12 10.x` and `mlir-tensorrt-{compiler,runtime} …+cuda12.trt109`.
Found the hard way — a plain `pip install nvtripy` here produced:

```txt
Successfully installed ... mlir-tensorrt-compiler-0.1.43+cuda12.trt109 mlir-tensorrt-runtime-0.1.43+cuda12.trt109
                          numpy-1.26.0 nvtripy-0.1.7 tensorrt-cu12-10.16.1.11 ...

>>> import tensorrt; tensorrt.__version__
'10.16.1.11'          # was 11.1.0.106
>>> import numpy; numpy.__version__
'1.26.0'              # was 2.1.0
```

Every other cookbook example silently moved to TensorRT 10.16 and NumPy 1.26. So `main.py` puts
nvtripy in a **private virtual environment** and runs the real example (`tripy_cases.py`) inside it:

```txt
cookbook interpreter : TensorRT 11.1.0.106
.venv                : TensorRT 10.16.1.11 (nvtripy 0.1.7), from .../nvtriPy/.venv/lib/python3.12/site-packages/tensorrt/
```

Two TensorRT versions in one container, neither disturbing the other. The first run creates the
venv and needs network access; without it the example prints `Skipped` and exits 0.

### The venv is not a workaround waiting to be removed

It would be easy to read the above as "the resolver picked badly, pin harder and it will be fine".
It is not. `report_version_matrix()` reads nvtripy's **own** metadata, and the bound is declared
upstream:

```txt
nvtripy 0.1.7 runtime requirements, as declared upstream:
    tensorrt-cu12<11,>=10.15
    mlir-tensorrt-compiler==0.1.43+cuda12.trt109
    mlir-tensorrt-runtime==0.1.43+cuda12.trt109
    colored==2.2.3
    nvidia-cuda-runtime-cu12
```

`<11` is nvtripy refusing TensorRT 11 by declaration — there is no pin that makes it coexist with
the cookbook's interpreter. `cu12` is the second half: the package index publishes only `cuda12`
builds, up to `trt109`, so in a CUDA 13 container there is no artifact that would fit even if the
TensorRT bound were relaxed. Hence the two columns:

```txt
component                            cookbook                  .venv
TensorRT                           11.1.0.106             10.16.1.11
NumPy                                   2.1.0                 1.26.0
mlir-tensorrt-compiler                      -   0.1.43+cuda12.trt109
nvidia-cuda-runtime                         -                12.9.79
```

A CUDA 12 stack runs at all because the wheels **ship their own CUDA runtime**
(`nvidia-cuda-runtime-cu12` above) and the driver is backward compatible; the container's CUDA 13
toolkit is never used by the venv. Only the GPU driver is shared.

Two guards keep this from rotting silently, because "an old pin that still installs" looks exactly
like "a pin that is still correct":

+ the `tensorrt-cu12<11` requirement is **asserted**, so the day upstream drops it the example fails
  loudly and tells you to re-check whether the venv is still needed at all;
+ the resolved stack is compared against `TESTED_STACK`, and `NVTRIPY_VERSION` against the newest
  release on the index, so both dependency drift and a stale pin are reported.

Currently `0.1.7` is the newest release on both the NVIDIA index and PyPI, and the assert holds.

## The cases

### 1. Eager, then compiled — and they are not identical

The selling point: the same `tp.Module` object runs both ways.

```python
model = MLP(4, 8, 2)
eager = model(x)                                              # runs now, values available now
executable = tp.compile(model, args=[tp.InputInfo(shape=(3, 4), dtype=tp.float32)])
compiled = executable(x.eval())
```

Anything not declared an `InputInfo` becomes a **compile-time constant**, which is why the
`Executable` takes one argument (`args0`) and not the weights.

But the two do not agree exactly:

```txt
eager    : [0.41942882537841797, ...]
compiled : [0.4194698929786682, ...]
max |eager - compiled| = 4.107e-05  (9.8e-05 relative)
```

Localizing it: an elementwise-only graph (`gelu` then `*2`) is **bit-identical**, while a single
`tp.Linear` is not — and it is **eager** that drifts, not the compiled engine:

```txt
one Linear, exact answer 0.28: eager 0.28000974655151367, compiled 0.2800000011920929
```

So the matmul is the difference, at roughly TF32 magnitude. Worth knowing before you chase a small
numeric discrepancy in eager mode: eager is for checking *shapes and logic*, not for validating the
last digits of what you are about to deploy.

### 2. Everything is lazy, which breaks the obvious way to time it

A tensor is computed only when something needs its value:

```txt
defining the tensor :    2.070 ms
first .eval()       :  107.917 ms  <- compiles, then executes
second .eval()      :    0.012 ms  <- already evaluated, cached
```

Timing the definition measures graph construction and nothing else. The first use pays for the
TensorRT build (eager mode has no eager execution underneath — it compiles too), and after that
the value is cached.

The same laziness shows up at the `Executable` boundary, which **rejects** an unevaluated input
rather than evaluating it for you:

```txt
TripyException: Hint: Try calling `.eval()` on the tensor.
```

### 3. Dynamic shapes are TensorRT optimization profiles

`InputInfo` takes `(min, opt, max)` per dimension — the same idea as `IOptimizationProfile`:

```python
tp.InputInfo(shape=((1, 4, 8), 4), dtype=tp.float32)   # dim0 in 1..8 tuned at 4, dim1 fixed at 4
```

One `Executable` then serves batch 1, 4 and 8. Batch 9 raises, and the message underneath is
TensorRT's own, naming the profile it failed against:

```txt
IExecutionContext::setInputShape: ... Set dimension [9,4] for tensor arg0 does not satisfy any
optimization profiles. Valid range for profile 0: [1,4]..[8,4].
```

Out of range you get an error, not a silent rebuild.

### 4. The `Executable` is the deployable artifact

```txt
compile :    423.5 ms
load    :      2.5 ms   (89 KiB on disk)
```

`executable.save(path)` / `tp.Executable.load(path)` — **168x** cheaper than compiling, with
identical output (the ratio moves run to run; the order of magnitude is the point). This is Tripy's equivalent of serializing an engine.

## What is not here

The [upstream examples](https://github.com/NVIDIA/TensorRT-Incubator/tree/main/tripy/examples)
also cover ResNet50, NanoGPT, Stable Diffusion, SAM2 and ModelOpt quantization. All of them need
gated or multi-GB Hugging Face downloads plus `torch`/`transformers` **inside the venv**, on top of
a frontend whose API is still pre-1.0. They are worth reading upstream; porting them here would be
maintenance for a moving target.

## Related

+ [`../../06-DLFrameworkTRT/Torch-TensorRT/`](../../06-DLFrameworkTRT/Torch-TensorRT/README.md) —
  the other "write it in Python, get TensorRT" frontend, mature and covered in depth.
+ [`../../08-Advance/MultiOptimizationProfile/`](../../08-Advance/MultiOptimizationProfile/README.md)
  — the profile mechanism of case 3, at the TensorRT API level.
+ [`../TritonServerDeploy/`](../TritonServerDeploy/README.md) — the other example here that has to
  install a whole toolchain of its own before it can run.
