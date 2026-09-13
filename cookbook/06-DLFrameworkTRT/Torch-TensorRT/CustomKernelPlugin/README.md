# Custom Kernel Plugin (QDP)

+ Run a custom Triton kernel *inside* the TensorRT engine instead of falling back to PyTorch.

+ Steps to run.

```shell
python3 main.py
```

Requires `triton` (present in the NGC PyTorch image) and TensorRT >= 10.7 for the
Quick Deployable Plugin system.

## The structural win

A `torch.library` custom op has no TensorRT converter, so the partitioner cuts
around it:

```
no plugin   : segments ['_run_on_acc_0', '_run_on_gpu_1', '_run_on_acc_2']
QDP plugin  : segments ['_run_on_acc_0']
```

```python
torch_tensorrt.dynamo.conversion.plugins.generate_plugin("cookbook::scale_mul")
torch_tensorrt.dynamo.conversion.plugins.generate_plugin_converter(
    "cookbook::scale_mul", supports_dynamic_shapes=True, requires_output_allocator=False)
```

Neither call sees the kernel source. `generate_plugin` drives the **meta kernel**
under `FakeTensorMode` to derive a symbolic shape descriptor, and wraps the eager
op as the plugin's runtime implementation — which is why
`@torch.library.register_fake` has to be correct, not merely present.

## …is not a speed win

```
no plugin  : 3 segments, 0.281 ms
QDP plugin : 1 segment , 0.706 ms  (2.51x slower)
```

The generated plugin is a **JIT** plugin: at engine runtime TensorRT hands
control back to Python, which runs the Triton kernel through PyTorch and copies
the result into TensorRT's output buffer. That callback happens on **every**
inference, while the graph break it replaced cost only two boundary crossings of
an already-launched graph.

On this model the callback is the more expensive of the two. Upstream's example
states that the plugin approach "avoids that overhead entirely" but never
measures it; here it is measured, reproducibly, in both orders.

The structural win is still real and can matter more than the latency:

+ one engine, so the whole model can be serialized and shipped as one artifact;
+ no PyTorch segment, so [`../CudaGraphs/`](../CudaGraphs/README.md) can capture it;
+ no Python in the *graph*, only in the plugin callback.

**The fix for the latency is an AOT plugin** (`aot_plugin.py` / `nvrtc_aot_plugin.py`
in the upstream repo): the kernel is compiled to PTX and embedded in the engine,
so there is no Python at runtime at all. Not covered here.

## Rule

Measure before adopting. A JIT plugin buys structure, not speed.

## Related

+ [`../ConverterOverloading/`](../ConverterOverloading/README.md) — writing a converter by hand, when the op maps onto existing TensorRT layers.
+ [`../TorchCompileBackend/`](../TorchCompileBackend/README.md) — where `_run_on_acc_*` / `_run_on_gpu_*` come from.
+ `05-Plugin/` — the lower-level `IPluginV3` route, independent of PyTorch.
