# Onnx Runtime

+ A cross-platform inference engine for ONNX models, and the reference a TensorRT result is checked against.

+ Repository [Link](https://github.com/microsoft/onnxruntime) ·
  [Document](https://onnxruntime.ai/docs/)

+ Steps to run.

```bash
python3 main.py
```

Measured with **onnxruntime-gpu 1.24.4** and **TensorRT 11.1.0.106** on a B200. That ORT version is
not a choice: `pip install -e .` pins it, because `nvidia-modelopt[onnx]` requires
`onnxruntime-gpu~=1.24.2`.

## Why a TensorRT cookbook has an ONNX Runtime directory

Not because ORT is a competitor. Because it is the **golden output you check a TensorRT engine
against**, and because its Execution Provider mechanism is the thing people most often confuse with
TensorRT itself. So this example is about the seams between the two.

## 1. The Execution Provider you asked for is not the one you got

`get_available_providers()` lists what the **wheel was compiled with**, not what can run on this
machine. Ask for a provider whose dependencies are missing and ORT drops it, uses CPU, and prints a
warning — it does **not** raise:

```txt
onnxruntime 1.24.4, get_device() = GPU
compiled with: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
this container has TensorRT 11.1.0.106

TensorrtExecutionProvider    -> silently fell back: got ['CPUExecutionProvider']
CUDAExecutionProvider        -> silently fell back: got ['CPUExecutionProvider']
CPUExecutionProvider         -> ok
```

**A script that "uses the TensorRT EP" can be running entirely on the CPU and still produce correct
answers, only slower.** `session.get_providers()` is the ground truth — assert on it. This is the
single most common way an "ORT with the TensorRT EP is slow" report gets produced.

Note the failure has two shapes. A provider whose library will not load is swapped out at **session
creation**; one that loads but has no kernels for this GPU gets all the way to **run** time and
raises there. Both were seen on this machine — ORT 1.29.0 reached run time and raised
`cudaErrorNoKernelImageForDevice`, while 1.24.4 cannot even load the library. Detecting both means
creating the session, running it, *and* then checking `get_providers()`.

### Why the TensorRT EP did not load here

ORT's message blames PATH / `LD_LIBRARY_PATH` / GPU support, which sends people looking in the wrong
place. The real reason is in the provider library's own dependencies:

```txt
Why: libonnxruntime_providers_tensorrt.so cannot resolve its own dependencies.
    libcublas.so.12 => not found
    libnvinfer.so.10 => not found
    libnvonnxparser.so.10 => not found
    libcudart.so.12 => not found
Two independent major-version mismatches, and the sonames are not compatible across either:
    TensorRT - the wheel wants `.so.10`, this container provides libnvinfer.so.11
    CUDA     - the wheel is a `cu12` build, this container is CUDA 13.3
```

**Two independent major-version mismatches**, either of which alone would be fatal: the wheel is
built against TensorRT 10 *and* against CUDA 12, while this container is TensorRT 11 and CUDA 13.
Sonames are not compatible across a major version, so no amount of PATH fixing produces a `.so.10`
or a `.so.12` that is not installed.

**Nor is upgrading ORT the answer.** The version is pinned by `nvidia-modelopt[onnx]`
(`onnxruntime-gpu~=1.24.2`), and newer wheels are still `cu12` and still TensorRT 10 anyway. On this
stack the GPU providers need a source build of ORT.

This affects only ORT's *embedded* use of TensorRT. TensorRT itself, and every other example in the
cookbook, is unaffected — `case_reference_for_tensorrt` below builds a real engine in this same
process.

## 2. The plain path

Session I/O metadata comes straight out of the ONNX file, symbols included:

```txt
Input   0: x, ['nBS', 1, 28, 28], tensor(float)
Output  0: y, ['nBS', 10], tensor(float)
Output  1: z, ['nBS'], tensor(int64)
```

(This example used to contain `for name, tensor in output_name_list, output_list:`, which iterates
over a 2-tuple of lists and prints the two output *names* as if they were a name/value pair. It is
`zip` now.)

## 3. ORT as the reference — and TF32

Both ORT and TensorRT consume the same ONNX file, so a disagreement is TensorRT's *build choices*
rather than a modelling difference. The size of it surprises people:

```txt
default (TF32 allowed)   max|ORT - TRT| = 6.146e-03  (3.39e-04 relative), argmax equal: True
TF32 cleared             max|ORT - TRT| = 9.537e-06  (5.26e-07 relative), argmax equal: True
```

**Clearing one flag moved the disagreement by 644x.** `BuilderFlag.TF32` is on by default, so a
TensorRT engine built with no precision flags at all is *not* doing FP32 matmuls — it is doing TF32
ones on the tensor cores, with a 10-bit mantissa.

So before blaming a conversion bug for a 1e-3 discrepancy against ORT: clear TF32 and re-measure. If
the gap collapses, there is no bug, and the remaining ~1e-5 is ordinary floating-point
reassociation. If it does not, now you have a real lead.

## 4. ORT rewrites the graph, and the rewrite is not portable

`optimized_model_filepath` writes out what ORT will actually execute — the best way to see a
runtime's own fusions:

```txt
original : 12 nodes ['Conv', 'Relu', 'MaxPool', 'Conv', 'Relu', 'MaxPool', 'Reshape', 'Gemm', 'Relu', 'Gemm', 'Softmax', 'ArgMax']
optimized: 10 nodes ['Conv', 'MaxPool', 'Conv', 'MaxPool', 'ReorderOutput', 'Reshape', 'FusedGemm', 'Gemm', 'Softmax', 'ArgMax']
domains now in use: ['com.microsoft', 'com.microsoft.nchwc']
TensorRT on the optimized file: ok=False, creator && "Plugin not found, are the plugin name, version, and namespace correct?"
```

The three `Relu`s are folded into their producers, and `FusedGemm` + `ReorderOutput` appear. **Do
not feed this file to TensorRT.** `com.microsoft.nchwc` ops are ONNX Runtime's private
layout-aware kernels; to TensorRT they are unknown ops, so it reports them as missing plugins.

Dump the optimized model to **read** it. Always build TensorRT from the original.

## 5. Per-node timings, with no external tool

```txt
ort_profile_....json: 122 events, 100 node kernel timings
    Conv                468 us  35.7%
    FusedGemm           415 us  31.7%
    MaxPool             185 us  14.1%
    ReorderOutput        66 us   5.0%
    Gemm                 59 us   4.5%
```

The op names are the **optimized** ones from case 4, which is how you know this is what really ran.
The file is Chrome-trace format — open it in `chrome://tracing` or Perfetto for a timeline.

## 6. One session, any batch size

```txt
batch   1: y(1, 10), z(1,)
batch   4: y(4, 10), z(4,)
batch   8: y(8, 10), z(8,)
batch  37: y(37, 10), z(37,)
```

37 works as readily as 1, with no optimization profile and no rebuild. That flexibility is exactly
the freedom TensorRT trades away for the ability to pick kernels and memory layouts for a known
shape range — compare
[`../../08-Advance/MultiOptimizationProfile/`](../../08-Advance/MultiOptimizationProfile/README.md),
where a shape outside the declared profile is an error.

## 7. `IOBinding` — place the buffers yourself

`session.run()` takes numpy arrays and returns numpy arrays, i.e. a copy in and a copy out on every
call. On a GPU provider those are host-device transfers that can dominate a small model.
`IOBinding` + `OrtValue` is ORT's answer, and the same idea as binding device pointers with
`IExecutionContext::setTensorAddress` in TensorRT.

On this machine the only working provider is CPU, so there is nothing to save; the win appears when
the provider is on a device and the caller already has the data there.

## A note on the log noise

ORT logs one `pthread_setaffinity_np failed` line **per thread** on a machine whose CPU affinity it
cannot set — over 100 red lines before any output. Its own message says how to stop it, and the fix
is real: set `SessionOptions.intra_op_num_threads` explicitly. Every session here does.

## Related

+ [`../ONNX/`](../ONNX/README.md) — the `onnx` package itself, including
  `onnx.reference.ReferenceEvaluator`, a *third* opinion for when ORT and TensorRT disagree.
+ [`../Polygraphy/`](../Polygraphy/README.md) — automates exactly the case-3 comparison
  (`polygraphy run --trt --onnxrt`), across every intermediate tensor rather than just the outputs.
+ [`../../04-Feature/`](../../04-Feature/README.md) — the TensorRT precision flags that case 3
  turns out to depend on.
