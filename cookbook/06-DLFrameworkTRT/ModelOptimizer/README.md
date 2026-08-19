# Model Optimizer

+ FP8 and INT8 post-training quantization with ModelOpt, compiled by the Torch-TensorRT Dynamo frontend.

+ Steps to run.

```bash
python3 main.py
```

The path is **PyTorch model → `mtq.quantize` inserts Q/DQ → `torch_tensorrt.dynamo.compile` builds a
strongly-typed engine.** Since TensorRT 11 removed weak-typing INT8 calibration, this is the
sanctioned way to get a quantized engine out of PyTorch: the Q/DQ nodes are what tell the builder
where the precision changes.

Measured on B200, TensorRT 11.1.0.106, `torch_tensorrt` 2.14.0a0, `nvidia-modelopt` 0.44.0,
small CNN (4 conv + classifier), batch 32:

| precision | quantizers | eager max abs diff | TensorRT max abs diff | latency |
| --- | ---: | ---: | ---: | ---: |
| FP32 | – | – | – | 0.079 ms |
| FP8 | 19 | 2.687e-03 (3.46%) | 2.653e-03 | 0.088 ms |
| INT8 | 19 | 8.919e-04 (1.15%) | 8.676e-04 | 0.084 ms |

## Read the eager column first

The quantized module is **still a PyTorch module** — it runs eagerly before TensorRT is involved at
all. That makes the eager column the quantization error *alone*, and it is the right place to find a
calibration problem: if the eager error is already unacceptable, no amount of engine debugging will
help.

The TensorRT column then answers a different question: did the builder honour the Q/DQ, or did it
re-invent the precisions? Here TRT tracks eager closely (2.653e-03 vs 2.687e-03), which is what a
correctly honoured Q/DQ graph looks like. A TensorRT error much *smaller* than eager would mean the
quantization was quietly dropped.

Note the model is small enough that quantization does not pay for itself in latency (0.088 vs
0.079 ms) — the point here is the mechanism and the numerics, not a speed-up.

## Calibration is a loop, not a config

`mtq.quantize(model, config, forward_loop)` calls `forward_loop` with the model, and whatever it
pushes through sets the amax values. Unrepresentative data produces a quantized model that is wrong
in a way no shape check will find. The calibration set here is synthetic but **fixed** (seeded), so
the amax values — and therefore the engine and the numbers above — are reproducible.

## Two things that bite

### The export needs ModelOpt's context manager

`torch.export.export` on a quantized module fails with:

```txt
RuntimeError: We found a fake tensor in the exported program constant's list.
```

The message names neither ModelOpt nor quantization. The quantizers hold fake tensors that only
resolve in export mode, so the export must happen inside
`modelopt.torch.quantization.utils.export_torch_mode()`. Without it, both FP8 and INT8 fail at this
step and it looks like a Torch-export bug.

### `use_explicit_typing` + `enabled_precisions` is **accepted**, not rejected

Coming from TensorRT 8/10 habits the instinct is to *ask* for a precision with `enabled_precisions`;
the strongly-typed way is to put Q/DQ in the graph and let it speak. Documentation for other
versions says passing both is an error.

**In `torch_tensorrt` 2.14.0a0 it is accepted silently.** That is worse than a rejection — the two
settings can disagree with nothing to tell you. Pass one or the other, and do not rely on being
warned.

## Why not ViT / CIFAR

Upstream (`quantize_vit_fp8.py`, `vgg16_ptq.py`) uses a timm ViT and CIFAR downloads. The cookbook
does not download at run time, and what matters here is the *structure* — conv stack, pooling,
classifier — plus a reproducible calibration set. See
[`../../03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/`](../../03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/README.md)
for the ONNX-route equivalent.

## Related

+ [`../../03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/`](../../03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/README.md)
  — the same idea via ONNX rather than the Dynamo frontend.
+ [`../../02-API/Layer/QDQStructure/`](../../02-API/Layer/QDQStructure/README.md) — what Q/DQ
  placement means at the network level, including block quantization.
+ [`../../07-Tool/OnnxFP8QDQConvert/`](../../07-Tool/OnnxFP8QDQConvert/README.md) — converting
  Transformer-Engine's custom FP8 Q/DQ into the standard operators.
