# Triton AOT Plugin

+ Compile an **OpenAI-Triton** kernel *ahead of time* and ship it inside a C++ `IPluginV3`.

Then let a generator write the plugin for you, and let the TensorRT builder pick among the AOT
variants.

+ Steps to run.

```bash
python3 main.py
```

All numbers re-measured on **B200** 2026-09-06 (TensorRT 11.1.0.106, Triton 3.7.1, CUDA 13.3,
1965 MHz) on `[16, 512, 512]`.

## Why AOT, when the cookbook already runs Triton in a plugin

[`../PythonPlugin/add_scalar_triton.py`](../PythonPlugin/README.md) puts a
Triton kernel in a plugin the easy way: a Python plugin calling Triton's **JIT**. That is the right
tool while you are still writing the kernel. It is the wrong tool for deployment — the serving
process needs Python, Triton and a compiler, and the first `enqueue` pays for compilation.

`triton.tools.compile` takes the other road: it emits **C source with the cubin embedded as a byte
array**, plus a `cuLaunchKernel` wrapper. `triton.tools.link` then gives those launchers stable
symbol names. What ships is one `.so`:

```txt
AddScalarTritonPlugin.so NEEDED: libcudart.so.13, libcuda.so.1, libstdc++.so.6, libgcc_s.so.1, libc.so.6
```

No `libpython`, no Triton — and `libnvinfer` is not there either, because a plugin only implements
interfaces that TensorRT resolves when it loads the library. Case 3 does not stop at reading `ldd`:
it rebuilds and reruns the engine in a subprocess where `import triton` **raises**, and checks the
numbers.

## The pipeline

```txt
kernel.py ──triton.tools.compile──> add_scalar.<hash>_<spec>.c   (cubin as a byte array + launcher)
                                    add_scalar.<hash>_<spec>.h
          ──triton.tools.link─────> add_scalar.c / add_scalar.h  (stable names, one algo_id per variant)
          ──sed───────────────────> the fp32 ABI fix, see below
          ──nvcc──────────────────> AddScalarTritonPlugin.so     (161 KiB, cubin inside)
```

The hash in the per-variant file name changes whenever the kernel or the signature changes, which is
exactly why `triton.tools.link` exists and why a hand-written plugin must call the *linked* entry
point (`add_scalar_default`), never the hashed one.

## The trap: an `fp32` kernel argument is silently read as garbage

`triton.tools.compile` declares an `fp32` kernel argument as **`double`** in the generated C, and
then passes `&scalar` straight into the `cuLaunchKernel` argument array — where the kernel's
parameter slot is **4 bytes**. The kernel reads the low half of the double:

```txt
generated C prototype: CUresult k_...(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, double scalar, int32_t n_element);
as generated (double): launch returned 0, x + 1.0 = [0. 1. 2. 3. 4. 5. 6. 7.]
after the sed (float) : launch returned 0, x + 1.0 = [1. 2. 3. 4. 5. 6. 7. 8.]
```

`1.0` as a double is `0x3FF0000000000000`, whose low half is `0x00000000` — so the kernel computes
`x + 0`. **The launch returns `CUDA_SUCCESS`.** Any "round" value behaves the same way; `0.1` comes
out as `-1.59e-23`. Case 2 reproduces this with plain ctypes, no TensorRT involved, and asserts both
the broken and the fixed behaviour, so the day Triton fixes it this example fails loudly instead of
quietly keeping a pointless `sed`.

The fix is one line in the `Makefile` (and in `plugin_gen.py`, per declared attribute):

```makefile
sed -i 's/double scalar/float scalar/g' $(STEM).c $(STEM).h $(STEM).*.c $(STEM).*.h
```

Note the caller's declaration has to move with it — case 2 switches its `ctypes` argtype from
`c_double` to `c_float` at the same time. Half a fix is still wrong.

## Two more things that fail quietly

+ **A plugin library must export `setLoggerFinder` as well as `getCreators`.** Exporting only
  `getCreators` is not a link error and not a load error: `loadLibrary` returns a handle and simply
  never registers anything, so the failure appears much later and somewhere else as
  `Cannot find plugin: AddScalarTriton, version: 1`. An empty body is enough.
+ **The linked dispatcher enforces the alignment you hinted.** `*fp32:16` in the signature makes the
  generated dispatcher check `ptr % 16 == 0` and return `CUDA_ERROR_INVALID_VALUE` **without
  launching**. Ignore the return value and the output buffer keeps whatever was in it. `enqueue`
  checks it.

Two smaller observations, not acted on: the generated `add_scalar(..., algo_id)` bounds-checks with
`assert(algo_id < sizeof(kernels))` — `sizeof` of the array in bytes, not its length — and the
launcher has no `return` on the `gX * gY * gZ == 0` path.

## Generating the plugin instead of writing it

`AddScalarTritonPlugin.cu` is ~330 lines of which about six are about the kernel. Everything else
follows mechanically from "one input, one output, same shape, FP32, elementwise" — and it is the
part that fails silently. So `plugin_gen.py` writes it:

```python
GELU_SPEC = dict(
    plugin_name="GeluTriton", kernel_file="kernel.py", kernel_name="gelu_kernel", stem="gelu",
    n_input=1, attribute_list=[],
    signature="*fp32:16, *fp32:16, i32, {block_size}",
    grid="(n_element + {block_size} - 1) / {block_size}, 1, 1",
    variant_list=[dict(block_size=128, num_warps=1), dict(block_size=1024, num_warps=4), dict(block_size=4096, num_warps=8)],
)
```

```txt
generated 339 lines of C++ for GeluTriton from a 9-key spec, 3 AOT variants -> 3 TensorRT tactics
```

The operator is deliberately **not** `AddScalar`: a different kernel, a different argument list and
no attribute at all, so the generator is shown to generalise rather than to re-emit the file next to
it. Scope is stated in the module docstring and is small on purpose — element-wise,
`kernel(in_ptr..., out_ptr, <float attributes>, n_element, BLOCK_SIZE: constexpr)`, FP32.

## AOT variants become TensorRT tactics

This is the part that is worth more than the code generation. The generator AOT-compiles the same
kernel several times, links them into one dispatcher with an `algo_id`, and maps each `algo_id` onto
a **plugin tactic** (`getNbTactics` / `getValidTactics` / `setTactic`). The TensorRT builder then
times them and bakes the winner into the engine:

```txt
engine information for GeluTritonLayer_myl0_1: LayerType='custom_layer', TacticName=''
-> the 3 AOT variants are still tactics, but which one won is NOT observable here
   (no `TacticValue`/`PluginType` on a Myelin-absorbed plugin; measure instead of reading a field)
all tactics    : 0.0067 ms
only variant 1 : 0.0187 ms
-> letting the builder choose is 2.78x the fixed first variant
```

(Measured on an idle B200. The ratio has now been seen at 1.83x and 2.78x on H100 and 2.78x on
B200, with the absolute times moving underneath it — treat it as "clearly more than 1", not as a
constant. The case asserts `> 1.2` for that reason.)

That is Triton autotuning **moved to build time**: no autotuner in the deployed process, no warm-up
cost, and the result travels inside the engine.

### The chosen tactic is no longer readable from the engine information

This example used to print `builder timed 3 tactics and kept 0x0000000000000003 -> BLOCK_SIZE=4096,
num_warps=8`, read out of `TacticValue` on the layer whose `PluginType` was `GeluTriton`. **On
TensorRT 11.1.0.106 that no longer works**, and the old code died with an `IndexError` because the
match returned nothing.

The plugin is absorbed into **Myelin**. What comes back is one layer named
`GeluTritonLayer_myl0_1` with `LayerType: custom_layer`, carrying **no `PluginType` key, no
`TacticValue` key**, and an empty `TacticName` — at `profiling_verbosity = DETAILED`, and equally
for a static-shape build, so it is not a side effect of the dynamic profile. `get_layer_information`
per layer and the `ONELINE` format show the same thing.

So the tactic search is still happening — it is just no longer *inspectable*. The example therefore
leans on the measurement instead of on a JSON field: build a second engine with only the first
variant available and time both. If the builder were not really timing the variants, the two engines
would run at the same speed. That is a better proof anyway, and it is the reason the case now
asserts on the speed-up rather than on a key existing.

`build_and_run` still sets `profiling_verbosity = DETAILED`, since that is what would expose the
tactic if a future TensorRT stops folding plugins into Myelin.

Tactic values must be **positive**, so tactic `n` is `algo_id n-1`, and `setTactic(0)` (TensorRT's
"just use the default") is mapped back to the first variant.

## Related

+ [`../PythonPlugin/`](../PythonPlugin/README.md) — the same Triton kernel
  the JIT way, plus NVRTC / CuPy / Numba / PyTorch backends. Start there, come here to deploy.
+ [`../CuteDSLPlugin/`](../CuteDSLPlugin/README.md) — a kernel in CUTLASS's
  Python DSL, also JIT.
+ [`../BasicExample/`](../BasicExample/README.md) — the `IPluginV3`
  boilerplate this one mirrors, with a hand-written CUDA kernel.
+ [`../Tactic+TimingCache/`](../Tactic+TimingCache/README.md) — plugin
  tactics on their own, and how the timing cache interacts with them.
