# Sparsity

+ Example of enabling sparse weights in TensorRT to reduce compute cost on supported hardware.

## Steps to run

```shell
python3 main.py
```

## Refit: sparse in, dense out? (`refit_sparse_vs_dense.py`)

A 2:4-sparse kernel reads two values per group of four plus a metadata index saying which two, so
refitting such an engine with **dense** weights is not a matter of copying more bytes — the kernel
cannot use them. That makes it the rare case where refit has a *semantic* constraint rather than a
shape one. The three possible outcomes are: refused (safe), accepted-and-wrong (dangerous), or
accepted-and-correct (no sparse kernel was ever selected).

Measured on B200, TensorRT 11.1.0.106, a `[32, 32, 3, 3]` FP32 convolution:

```txt
sparse weights, flag on    tactic cutlass3x_sm100_tensorop_s256x128x8tf32implicit_gemm
sparse weights, flag off   tactic cutlass3x_sm100_tensorop_s256x128x8tf32implicit_gemm
tactic identical with and without SPARSE_WEIGHTS: True
```

**The builder declined sparsity**, so the answer here is the third one: refit with dense weights is
accepted and gives results bit-identical to a freshly built dense engine (`max |diff| = 0.000e+00`).

That is an honest negative result, not a demonstration of the constraint. `SPARSE_WEIGHTS` is
**permission, not instruction** — a sparse kernel is used only when it is both allowed and faster,
and on this shape and architecture it was not chosen. Widening to 256 channels and FP16 did not
change it either: the tactic name stayed byte-identical with the flag on and off.

The lesson worth taking is the method: **compare tactic names, not plan sizes.** The plan differs by
a few bytes just from recording the flag, so plan size cannot tell you whether sparsity was used.
`profiling_verbosity = DETAILED` plus `IEngineInspector` can.
