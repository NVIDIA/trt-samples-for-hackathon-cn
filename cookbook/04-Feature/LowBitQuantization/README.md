# Low-Bit Quantization

+ NVFP4, MXFP8 and INT4-AWQ from PyTorch: what the formats cost, and which of them actually reach a TensorRT engine.

+ Steps to run.

```bash
python3 main.py
```

Below 8 bits the question stops being "does it run" and becomes "what survives the round trip".
Measured on B200, TensorRT 11.1.0.106, `nvidia-modelopt` 0.44.0, `torch_tensorrt` 2.14.0a0, on a model
with **both** `Conv2d` and `Linear` layers:

| format | eager error | via ONNX | via Torch-TensorRT |
| --- | ---: | --- | --- |
| FP8 | 4.29% | failed | **ok** |
| INT8 | **0.79%** | failed | **ok** |
| NVFP4 | 11.51% | failed | failed |
| MXFP8 | 4.37% | failed | failed |
| INT4-AWQ | 6.73% | failed | failed |

## Only INT8 and FP8 reach the engine

Torch-TensorRT's converter is explicit about it:

```txt
ValueError: quantize converter currently only accept INT8 or FP8 based quantize
```

So on this stack the sub-8-bit formats quantize cleanly in PyTorch and then **stop**. That is the
single most useful thing to know before designing around NVFP4 or MXFP8 through this path — the
PyTorch-side success is not evidence that the engine will exist.

The ONNX route fails for *every* format, including the two that work via Torch-TensorRT:

```txt
Expected node type 'onnx::Constant' for argument 'amax' of node 'symbolic', got 'prim::P...
```

The legacy TorchScript exporter cannot represent ModelOpt's quantizer modules, and the dynamo
exporter falls back to the same path. Since the two legs fail for *different* reasons, running both
is what tells you whether to blame the format or the exporter.

## Bit width alone does not decide accuracy

**INT8 (0.79%) beats FP8 (4.29%) by 5.4x at identical bit width.** The integer format tracks a
well-behaved activation range better than the float one, which spends bits on an exponent this model
does not need.

But the honest counterweight: on this model the 4-bit formats are genuinely worse (NVFP4 11.51%),
so **block scaling does not rescue 4 bits here**. A small model with narrow, well-behaved activation
ranges is exactly the case where fine-grained scaling has least to offer; the MX formats earn their
keep on large models with heavy-tailed distributions. Do not read this table as a ranking of the
formats in general — read it as a demonstration that the ranking is model-dependent and has to be
measured.

## Compare the engine against the *eager* module, not against FP32

Two questions get conflated constantly:

+ **is this format accurate?** → eager quantized vs FP32 (the `eager error` column)
+ **did TensorRT honour the Q/DQ?** → engine vs the **eager quantized module**

The second is the one that catches a silently-dropped quantization, and it needs the eager module as
the reference. Here FP8 gives 5.120e-03 and INT8 8.939e-04 engine-vs-eager — small relative to their
own quantization error, i.e. the Q/DQ was honoured. An engine-vs-FP32 number mixes both effects and
cannot distinguish them.

## Why not a timm ViT

Upstream (`examples/torch_onnx/torch_quant_to_onnx.py`) downloads a timm ViT/Swin. The cookbook does
not download at run time, and the property that matters here is having **both** convolutions and
linear layers in one graph, since TensorRT does not treat them alike.

## Related

+ [`../../06-DLFrameworkTRT/ModelOptimizer/`](../../06-DLFrameworkTRT/ModelOptimizer/README.md) — the
  FP8/INT8 path in depth, including the `export_torch_mode()` requirement used here.
+ [`../../02-API/Layer/QDQStructure/`](../../02-API/Layer/QDQStructure/README.md) — block quantization
  at the network level, where FP4 and E8M0 hit their own limits.
+ [`../../02-API/Layer/DynamicQuantize/`](../../02-API/Layer/DynamicQuantize/README.md) — FP4 through
  `IDynamicQuantizeLayer`, which is the API that does support it.
