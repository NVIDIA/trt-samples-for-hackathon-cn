# Workflow of ModelOptimizer post-processing -> TensorRT

+ A workflow of: train a model in pyTorch, post-process it into a reduced-precision model with **NVIDIA TensorRT Model Optimizer (ModelOptimizer)**, then parse the resulting ONNX in TensorRT, build a **strongly-typed** engine and do inference.

+ Steps to run.

```bash
python3 main.py
```

+ The network used here is the same `class Net` as in `00-Data/get-model-part1.py` (a small MNIST CNN). Make sure the data files in `00-Data/data` (`TrainData.npz`, `TestData.npz`, `InferenceData.npz`) have been prepared (see `00-Data`). Training / evaluation use `TrainData.npz` / `TestData.npz`, while the final TensorRT inference uses `InferenceData.npz`.


+ This file merges the three ModelOptimizer post-training workflows into one, sharing the dataset, network, FP32 pre-training, ONNX export helper and the strongly-typed TensorRT build helper. Here are the independent cases:
  + `case_autocast`: convert the FP32 ONNX to a mixed **FP16/FP32** ONNX with `modelopt.onnx.autocast` (`model-fp16-autocast.onnx` / `.trt`), next to a FP32 baseline engine (`model-fp32.trt`) so the accuracy cost of the conversion is measured rather than assumed.
  + `case_autocast_exclude`: the same conversion steered with `op_types_to_exclude` (`model-fp16-autocast-excluded.onnx` / `.trt`).
  + `case_qat_train`: insert INT8 fake-quantizers into the *pyTorch* model with `modelopt.torch.quantization`, run **quantization-aware training (QAT)**, then export INT8 Q/DQ ONNX (`model-int8-qat.onnx` / `.trt`).
  + `case_onnx_post_train`: run ModelOptimizer **ONNX post-training quantization** (`modelopt.onnx.quantization`) to insert **FP8 (E4M3)** Q/DQ nodes (`model-fp8.onnx` / `.trt`).

## The three ModelOptimizer entry points

| Case                   | ModelOptimizer module            | Operates on   | Result precision | Calibration            |
| ---------------------- | -------------------------------- | ------------- | ---------------- | ---------------------- |
| `case_autocast`        | `modelopt.onnx.autocast`         | ONNX graph    | FP16 / FP32 mix  | none (pure cast)       |
| `case_qat_train`       | `modelopt.torch.quantization`    | pyTorch model | INT8             | amax + QAT fine-tuning |
| `case_onnx_post_train` | `modelopt.onnx.quantization`     | ONNX graph    | FP8 (E4M3)       | absolute-max on data   |

+ **AutoCast** rewrites a FP32 ONNX into a mixed-precision graph by inserting explicit `Cast` nodes, keeping numerically-sensitive nodes in FP32. No calibration data is needed.

+ **QAT** (`modelopt.torch.quantization`) inserts fake-quant modules into the `torch.nn.Module`, initializes their `amax` with a short calibration pass, then keeps training with the fake-quantizers in place so the weights adapt to the quantization noise. ModelOptimizer emits standard INT8 `QuantizeLinear` / `DequantizeLinear` pairs on export.

+ **ONNX post-training quantization** (`modelopt.onnx.quantization`) quantizes an already-exported FP32 ONNX graph directly. It is exporter-independent and robust, and matches the "ModelOptimizer -> ONNX -> TensorRT" deployment path. `quantize_mode="fp8"` uses absolute-max (`"max"`) calibration to collect the amax of each quantized tensor.

## Reading the AutoCast output

`report_precision()` counts where AutoCast actually put each tensor, because the interesting number
is not "is the model FP16" but how much of it stayed FP32 — AutoCast keeps numerically-sensitive
nodes in high precision and pays for each switch with a `Cast` node:

```txt
model-fp32.onnx:                    13 nodes (0 Cast), initializers={'FP32': 8}
model-fp16-autocast.onnx:           15 nodes (2 Cast), initializers={'FP16': 8}
model-fp16-autocast-excluded.onnx:  18 nodes (5 Cast), initializers={'FP32': 4, 'FP16': 4}
```

ModelOptimizer reports the same thing as a ratio: `Converted 13/13 nodes (100.00%) to fp16` for the
unrestricted conversion versus `Converted 11/13 nodes (84.62%)` once `Gemm` / `MatMul` are excluded.
Excluding operators is not free — the graph goes from 2 to 5 `Cast` nodes, because every boundary
between an FP16 region and an FP32 region needs one.

`case_autocast` then compares the mixed-precision engine against the FP32 baseline:

```txt
[check]:True,maxAbsDiff=6.019e-03,meanAbsDiff=2.391e-03,maxRelDiff=2.675e-03,meanRelDiff=8.451e-04
Predicted label unchanged: True
```

`y` (the logits) drifts by a few tenths of a percent relative, while `z` (the predicted label) is
bit-identical — which is the outcome a mixed-precision conversion is supposed to produce. The exact
figures move from run to run because the model is retrained each time; only the order of magnitude
and `Predicted label unchanged: True` are meaningful.

## When to use the exclusion knobs

Automatic node selection is a heuristic. When one operator turns out to be the one losing accuracy,
`op_types_to_exclude` / `nodes_to_exclude` pin it back to FP32 without giving up low precision
everywhere else. The related knobs `data_max`, `init_max` and `calibration_data` control the
magnitude thresholds AutoCast uses to decide that a node is unsafe to convert.

## BF16, and what actually decides that a node stays FP32

`case_autocast_bf16` and `case_node_sensitivity` extend the two FP16 cases above.

### BF16 is not a safer FP16

Both formats have 8 exponent bits; BF16 spends 7 bits on the mantissa where FP16 spends 10. So BF16
cannot overflow where FP16 can, and is **less precise wherever both fit**. AutoCast raises the opset
from 19 to **22** for it (13 is enough for FP16).

| | max &#124;logit − FP32&#124; |
| --- | --- |
| FP16 | 5.912e-03 |
| BF16 | 3.542e-02 |

BF16 is **6.0x further from FP32** here, and the predicted label is unchanged either way. On a model
that never approaches FP16's range limit, the three mantissa bits BF16 gives up buy nothing. Reach
for BF16 when activations have training-like dynamic range — not as a drop-in "safer FP16".

### `data_max` is the knob that decides "sensitive"

AutoCast keeps a node in FP32 when its I/O magnitudes exceed `data_max` (default 512). Sweeping it:

| `data_max` | FP16 init | FP32 init | Cast nodes | max &#124;logit − FP32&#124; | graph hash |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2 | 6 | 3 | 3.643e-03 | `8b437149` |
| 16 | 8 | 0 | 2 | 4.174e-03 | `db30de63` |
| 512 | 8 | 0 | 2 | 5.151e-03 | `db30de63` |
| 65504 | 8 | 0 | 2 | 5.912e-03 | `db30de63` |

Two things to read here, and the second is why the case hashes the graph.

**1. "Sensitive" is a statement about the threshold, not about the model.** This CNN's activations
peak near 3.3, so every threshold at or above 16 classifies every node identically; only `data_max=1`
moves the split.

**2. Rows 2–4 are the *byte-identical* graph (`db30de63`) and still report different errors.** That
spread — here monotonically increasing, on other runs not — is **TensorRT build nondeterminism, not
the knob**. Without the hash column, row 2 or 3 reads as a sweet spot worth tuning toward. It is not
one. *When a sweep changes a number, check whether it changed the artefact before explaining it.*

For the same reason `case_node_sensitivity` asserts the **Cast count** (deterministic: 3 vs 2) and
deliberately does **not** assert the error ordering — `data_max=1` has landed on both sides of the
others across runs. Pinning nodes to FP32 inside an FP16 graph is not free accuracy; each extra
boundary is another round trip through FP16.
