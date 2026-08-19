# ONNX FP8 Q/DQ Convert

+ Rewrite Transformer-Engine's custom FP8 Q/DQ operators into standard opset-19 `QuantizeLinear` / `DequantizeLinear`.

+ Steps to run.

```bash
python3 main.py
```

Transformer-Engine exports FP8 quantisation as **custom operators in the `trt` domain**, wrapped in
`Cast` nodes because the custom operator is float32-only:

```txt
Cast(fp16->fp32) -> TRT_FP8QuantizeLinear -> TRT_FP8DequantizeLinear -> Cast(fp32->fp16)
```

`scripts/convert_te_onnx_to_trt_onnx.py` in TensorRT-OSS rewrites that into standard opset-19 Q/DQ.
This example re-implements the conversion small enough to read, and needs no Transformer-Engine
installation: it synthesises a graph in exactly the shape TE emits, which is what makes the
before/after comparison verifiable instead of asserted.

Measured on B200, TensorRT 11.1.0.106, onnx 1.21.0, onnxruntime CPU.

| Model | ONNX nodes | TRT parses | Q/DQ layers | onnxruntime |
| ----- | ---------: | ---------- | ----------: | ----------- |
| TE custom operators | 8 | **yes** | 4 | **no** |
| opset19, `Cast` kept | 8 | yes | 4 | yes |
| opset19, `Cast` removed | 5 | yes | 4 | yes |

## The usual reason to run this no longer applies

The conversion is normally described as what makes a Transformer-Engine model loadable by
TensorRT. **On TensorRT 11.1 that is not true**: the parser already understands the custom
operators and maps `trt::TRT_FP8QuantizeLinear` straight onto `IQuantizeLayer`. Both the original
and the converted file build to the **same 4-layer engine**.

What the conversion is still for is **portability**. The `trt` domain is not a real ONNX domain, so
`onnx.checker` accepts the file (a custom domain is legal ONNX) while every other runtime refuses
it outright:

```txt
Fatal error: trt:TRT_FP8QuantizeLinear(-1) is not a registered function/op
```

After conversion the same model loads and runs in ONNX Runtime. If TensorRT is the only consumer,
the conversion buys nothing; if anything else has to read the model — a checker, a viewer, a second
runtime, a quantisation-debugging pass — it is the whole point.

## The four edits

1. `TRT_FP8QuantizeLinear` → `QuantizeLinear`, `TRT_FP8DequantizeLinear` → `DequantizeLinear`, and
   the node's `domain` back to `""`.
2. Add the zero-point input the standard operator requires and the custom one does not. **Its dtype
   is what selects FP8**: `TensorProto.FLOAT8E4M3FN`. The value is always 0 for the symmetric
   scaling TE uses.
3. Drop the `Cast` in front of Q and behind DQ. This changes the ONNX node count (8 → 5) but **not**
   the engine — 4 layers either way, because TensorRT folds the casts itself.
4. Bump the opset to 19. Not cosmetic: Q/DQ are much older, but **FP8 zero-point types are not**, so
   an opset-13 file carrying an E4M3 zero-point is invalid rather than merely old.

## The scale is the fiddly part

Edit 3 is where this stops being a search-and-replace, and it is worth doing once by hand to see why
the upstream script carries a `cast_scale` helper.

`QuantizeLinear` binds `x` and `y_scale` to the **same** type parameter `T1`. Removing the `Cast` in
front of Q leaves a float16 activation next to a float32 scale, and ONNX rejects it:

```txt
Type Error: Type parameter (T1) of Optype (QuantizeLinear) bound to different types
            (tensor(float16) and tensor(float))
```

Retyping the scale in place then fails a second time, in the opposite direction. The scale
initializer is **shared** between the Q and the DQ of a pair, and after dropping only some of the
casts the two ends legitimately need different types — on the weight branch the Q still sees a
float32 constant while the DQ now has a float16 consumer. So the fix is a **per-node copy** of the
scale, not a mutation of the shared one.

Both failures are invisible if you only test with TensorRT, which parses the broken file happily.

## Related

+ [`../../02-API/Layer/QDQStructure/`](../../02-API/Layer/QDQStructure/README.md) — what Q/DQ
  placement means at the network level.
+ [`../../03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/`](../../03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/README.md)
  — the sanctioned way to *produce* FP8 Q/DQ under TensorRT 11 strong typing.
+ [`../OnnxGraphSurgeon/`](../OnnxGraphSurgeon/README.md) — the general graph-rewriting toolbox.
