# PyTorch tensors

+ Feed a `TrtRunner` torch tensors instead of NumPy arrays.

+ Steps to run.

```bash
python3 main.py
```

## What comes back follows what went in

```
numpy in -> ndarray
torch in -> Tensor on cpu      (copy_outputs_to_host=True, the default)
torch in -> Tensor on cuda:0   (copy_outputs_to_host=False)
numpy and torch paths agree: True
```

`copy_outputs_to_host=False` leaves the result where the engine wrote it, so a
GPU-side next step needs no host round trip:

```
output stays on cuda:0, dtype torch.float32
argmax computed on the GPU: 8
```

The usual caveat still applies — the runner owns that buffer and overwrites it on
the next `infer()`, see
[`../04-ExtendInterop/`](../04-ExtendInterop/README.md).

## The trap: the first `infer()` decides the output container

```
fresh runner, torch first            -> Tensor
fresh runner, numpy first then torch -> DeviceView   <- same call, different type
DeviceView has `.device`? False
```

`copy_outputs_to_host=False` is documented to return a torch tensor when PyTorch
has GPU support, and a Polygraphy `DeviceView` otherwise. What is **not**
documented is that the choice is made from the **first feed dict the runner ever
saw**, not the current one.

A runner handed NumPy once keeps returning `DeviceView` afterwards, even when fed
torch tensors. `DeviceView` has no `.device` and no `.cpu()`, so the failure
surfaces as an `AttributeError` somewhere downstream rather than at the call that
caused it.

**Rule: one array type per runner.**

## The dtype argument

```
in torch.bfloat16 -> out torch.bfloat16, exact match = True
```

A BF16 network (built by hand, since precision is declared not requested — see
[`../05-BuildNetworkByHand/`](../05-BuildNetworkByHand/README.md)) driven
directly with `torch.bfloat16` tensors.

The usual justification is "NumPy does not support BF16". That is true of NumPy
itself, but be careful with the test:

```
numpy version: 2.1.0
np.dtype('bfloat16') resolves to bfloat16 -- registered by `ml_dtypes`, not by NumPy
ml_dtypes 0.5.4 is installed here
```

`np.dtype("bfloat16")` raises `TypeError` in a clean interpreter and starts
resolving once `ml_dtypes` is imported — which happens transitively in this
image. Code relying on it works by accident. `torch.bfloat16` needs no
registration, which is the robust reason to use torch tensors for BF16 work.

## Note on the upstream example

It pairs this feature with an INT8 `Calibrator`, which does not work on
TensorRT 11 at all — see
[`../06-Int8IsNowExplicit/`](../06-Int8IsNowExplicit/README.md). Only the tensor
half still applies.
