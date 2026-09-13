# Use cuFFT

+ Call cuFFT from a plugin, and give the ONNX `DFT` operator somewhere to go.

## Steps to run

```shell
make build
python3 main.py
```

TensorRT has no FFT layer, and its ONNX parser rejects `DFT` outright:

```txt
In node 0 with name:  and operator: DFT (checkDFT): UNSUPPORTED_NODE: false
```

So a spectral model stops at the parser. This closes that gap the same way
[`../UseCuBLAS/`](../UseCuBLAS/README.md) closes the GEMM one — a plugin that forwards to a CUDA
math library. `case_onnx_dft` shows both halves: the standard operator being refused, and the same
graph parsing once the node is moved into the `trt.plugins` domain.

Measured on B200, TensorRT 11.1.0.106, CUDA 13.3, n = 32, batch 4.

| Mode | input | output | reference | max abs diff |
| ---- | ----- | ------ | --------- | -----------: |
| C2C forward | `[.., n, 2]` | `[.., n, 2]` | `numpy.fft.fft` | 1.9e-06 |
| C2C inverse | `[.., n, 2]` | `[.., n, 2]` | `numpy.fft.ifft * n` | 1.4e-06 |
| R2C | `[.., n]` | `[.., n/2+1, 2]` | `numpy.fft.rfft` | 9.5e-07 |
| C2R | `[.., n/2+1, 2]` | `[.., n]` | `numpy.fft.irfft * n` | 3.8e-06 |
| ONNX `trt.plugins` node | `[.., n]` | `[.., n/2+1, 2]` | `numpy.fft.rfft` | 9.5e-07 |

ONNX has no complex type, so a complex tensor is a real tensor with a trailing dimension of 2 —
the convention the ONNX `DFT` operator itself uses. That is why the three modes change the tensor
**rank**, not just its extent, and why `getOutputShapes` has to branch on the mode.

## Three things that bite

### cuFFT does not normalise

A forward transform followed by an inverse multiplies the signal by `n`, and cuFFT leaves the
scaling to the caller. NumPy puts the `1/n` in the inverse, so `cufftExecC2C(..., CUFFT_INVERSE)`
equals `numpy.fft.ifft(...) * n`, and `cufftExecC2R` equals `numpy.fft.irfft(...) * n`. The example
prints both comparisons for the inverse case, because being wrong by an exact integer factor is a
recognisable signature worth seeing once.

### The plan belongs to the context, not to the plugin object

A cuFFT plan is tied to (transform type, length, batch), and the batch comes from the leading
dimensions, which are dynamic. So the plan is created in `onShapeChange` — called whenever the
runtime shapes change — rather than in the constructor or on every `enqueue`. The copy constructor
deliberately does **not** copy the plan handle: `clone()` is used by `attachToContext`, and two
contexts sharing one plan would be a use-after-free waiting for the first destructor.

### `cufftSetStream` is not optional

Without it cuFFT runs on the default stream. The numbers still come out right, because the driver
serialises everything, so the bug is invisible in a correctness test — the plugin just quietly
becomes a synchronisation point in the middle of the engine.

## Why not the upstream `fftPlugin`

TensorRT-OSS added `plugin/fftPlugin` in **11.2**, backing the ONNX `DFT` operator with cuFFT. This
machine runs **11.1.0.106**, where that plugin is not in the shipped library (`TopkLastDim` is
present, `FFT` is not), so it could not be verified here. Once 11.2 is installed, prefer the
upstream plugin — it handles multi-dimensional and axis-selectable transforms that this
deliberately small example does not. What stays useful here is the plumbing: cuFFT plan lifetime,
stream binding, the complex-as-trailing-dimension convention, and the `trt.plugins` domain trick.

## A `dlopen` trap worth knowing

`IPluginRegistry.load_library` passes the string straight to `dlopen`, which for a bare filename
searches the **system library paths and never the working directory**, and then returns `None`
rather than raising. The failure only surfaces much later as `Cannot find plugin: CuFFT`. Every
plugin example in this cookbook therefore writes `Path(__file__).parent / "X.so"`, and this one
does too.

## Related

+ [`../UseCuBLAS/`](../UseCuBLAS/README.md) — the same pattern with cuBLAS.
+ [`../ONNXParserWithPlugin/`](../ONNXParserWithPlugin/README.md) — parsing an ONNX file whose node
  resolves to a plugin.
+ [`../../07-Tool/OnnxGraphSurgeon/`](../../07-Tool/OnnxGraphSurgeon/README.md) — rewriting a
  standard operator into a plugin node in the first place.
