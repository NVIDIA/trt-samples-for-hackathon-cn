# CUDA Graphs

+ Replay a compiled module as a CUDA graph instead of launching kernel by kernel.

+ Steps to run.

```shell
python3 main.py
```

## Three ways to turn it on

```python
# scoped, preferred -- hands back the module to call
with torch_tensorrt.runtime.enable_cudagraphs(compiled) as cudagraphs_module:
    cudagraphs_module(data)

# for the whole session, convenient and easy to forget about
torch_tensorrt.runtime.set_cudagraphs_mode(True)
compiled(data)
torch_tensorrt.runtime.set_cudagraphs_mode(False)
```

All three paths give identical numbers.

## Whether it helps is a property of the model

CUDA graphs remove **launch** overhead. If the kernels are long enough to hide
their own launches, there is nothing to remove:

| model | plain | cudagraphs | gain |
| --- | ---: | ---: | ---: |
| resnet18, batch 16 | 0.627 ms | 0.616 ms | **1.02x** |
| resnet18, batch 1 | 0.255 ms | 0.235 ms | 1.09x |
| 40 small `Linear`, batch 1 | 0.205 ms | 0.104 ms | **1.97x** |
| 40 small `Linear`, batch 128 | 0.368 ms | 0.160 ms | **2.31x** |

Worth noting that the upstream tutorial demonstrates this feature on **ResNet18
at batch 16** — the row that gains nothing. TensorRT has already fused that model
into a handful of long kernels, so the launches were never the bottleneck. The
gain shows up on many-short-kernel models, and it is the *kernel count*, not the
batch size, that decides it (batch 128 gains as much as batch 1).

## One recording, tied to one shape

```
batch=64  :   2.201 ms   first call, records
batch=64  :   0.266 ms   replay
batch=128 :   0.728 ms   new shape, re-records
batch=128 :   0.230 ms   replay
batch=64  :   0.664 ms   back to 64, re-records again
```

The last line is the one to remember: returning to a shape that was recorded
before **re-records anyway**. There is no per-shape cache, only the most recent
recording. A workload that alternates between two shapes therefore pays the
recording cost on every call and is better off with CUDA graphs disabled.

## Graph breaks

```
segments: ['_run_on_acc_0', '_run_on_gpu_1', '_run_on_acc_2']
context returned CudaGraphsTorchTensorRTModule, not the module itself
plain 0.212 ms, cudagraphs 0.119 ms (1.78x)
```

When the module has breaks, the context manager returns a
`CudaGraphsTorchTensorRTModule` wrapper that records the **whole** sequence,
PyTorch segment included, so the breaks stop costing extra launches.

## Related

+ [`../TorchCompileBackend/`](../TorchCompileBackend/README.md) — where `torch_executed_ops` and the `_run_on_gpu_*` segments come from.
