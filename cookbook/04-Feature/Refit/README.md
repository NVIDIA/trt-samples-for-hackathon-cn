# Refit

+ Steps to run.

```bash
python3 main.py
```

+ Several methods to refit a TensorRT engine.

+ In case `case_dummy_engine`, we build a dummy engine with untrained weights.
  + Here we also show the usage of stripping weights engine, i.e., building a TensorRT engine wwithout weights inside.
  + We can see the size of stripping weights engine file (453 KiB) is much smaller than normal one (13 MiB) with weights inside.

+ In case `case_set_weights`, we refit the dummay engine with new weights, it contains two equivalent implementations.

+ In case `case_set_weights_gpu`, we refit the dummay engine with new weights, which GPU is on GPU.

+ In case `case_refit_engine`, we refit the dummy engine with trained ONNX file directly, it contains three equivalent implementations.

## `case_refit_only_what_changed` — refit the two weights that changed, not all eight

Adapted from `get_refit_weights()` in `exporter.py` and `Engine.refit_from_dict()` in
`utilities.py` of [Stable-Diffusion-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT)
(Apache-2.0). There the question is "the user switched to another LoRA — which weights changed?",
and the answer is to hash every ONNX initializer, hash the new state dict, and refit only the
mismatches. The idea is theirs; the code is rewritten for TensorRT 11 on MNIST weights.

The other cases here answer *how do I call the refit API*. This one answers *how much do I have to
push*. On a model of any size, moving every weight to change two of them is gigabytes of
host-to-device traffic for nothing.

```txt
    Refittable weights in the engine : 8
    Weights whose hash changed       : 2 -> ['gemm2.bias', 'gemm2.weight']
    Missing after setting only the changed ones: 0
    refit_cuda_engine() succeeded having set only the 2 changed weights
    Selective vs full refit, max abs diff: 0.000e+00
```

Hashing needs `np.ascontiguousarray` before `tobytes()` — the same discipline `case_set_weights`
calls a BLOODY lesson, for a different reason: `tobytes()` raises on a non-contiguous array, and a
C-order view of the same values hashes differently from an F-order one.

### Hash the dtype the engine stores, not the dtype on disk

The upstream casts every initializer to fp16 before hashing, because its engine is fp16. That is
not incidental. Take a weight and the same weight round-tripped through fp16 — a different fp32
array, but the same fp16 array:

```txt
    Same weights, one round-tripped through fp16:
        hashed as fp32 -> differs: True
        hashed as fp16 -> differs: False
```

An fp32 hash reports a change an fp16 engine could not represent, and you refit for nothing.

### Measured: a STRIP_PLAN engine will not tell you its weights are missing

The expectation going in was that a weight-stripped engine would report the six weights we did not
set as missing and refuse to refit. It does neither:

```txt
    STRIP_PLAN engine is 0.29 MB vs 13.39 MB with weights
    get_missing_weights() before setting anything: 0
    After setting only 2: 0 missing, refit_cuda_engine() -> True
    ... and the output is wrong by 1.972e+01
```

A 46x smaller engine, `get_missing_weights()` returns **0 before anything has been set**, and
`refit_cuda_engine()` returns **True**. The only signal that anything is wrong is the output.

So the selective technique applies to **updating an engine that already has weights**, and there is
no API that will stop you from applying it to one that does not. This is the same silent success
already recorded in [`../../07-Tool/trtexec/README.md`](../../07-Tool/trtexec/README.md) ("a
stripped engine loads, runs, and reports `PASSED` while producing all zeros") and
[`../WeightStripping/`](../WeightStripping/README.md), reached from a third direction.

Measured on TensorRT 11.3.0.99, B200, 2026-10-12.
