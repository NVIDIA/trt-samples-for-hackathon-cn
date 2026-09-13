# ONNX PTQ With Plugin

+ Quantize an ONNX graph that contains a **custom plugin op**, then build and run it.

+ Steps to run.

```bash
python3 main.py
```

Post-training quantization has to **run** the graph to observe activation ranges. A graph with a
custom operator cannot be run by ONNX Runtime — the op does not exist there — so ordinary PTQ stops
before it starts. That is the one intersection of plugins and quantization neither `05-Plugin/*` nor
[`../../03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/`](../../03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/README.md)
covers.

The plugin is the cookbook's own `AddScalarPlugin` from [`../BasicExample/`](../BasicExample/README.md),
so there is no new CMake project — which also shows the workflow works with a plugin you already have.

Measured on B200, TensorRT 11.1.0.106, `nvidia-modelopt` 0.44.0, ONNX Runtime 1.24.4.

## Result

```txt
FP32 graph with plugin      : built and ran
quantized graph with plugin : built and ran
INT8 vs FP32: max |diff| = 1.677e-02 (relative 2.13%)
```

## `trt_plugins=` is documented as the answer. It does not work here.

Two independent blockers, and **none of the three error messages points at its own cause** — which
is the reason this example spells them out:

### 1. ModelOpt reads an attribute TensorRT 11 removed

```txt
AttributeError: 'tensorrt.tensorrt.IPluginRegistry' object has no attribute 'plugin_creator_list'
```

`modelopt/onnx/trt_utils.py:117` uses `plugin_creator_list`; TensorRT 11 renamed it to
`all_creators`. The offending line is a **debug log statement that only counts creators** —
cosmetic in intent, fatal in effect. `apply_modelopt_trt11_shim()` restores the old name as a
read-only alias. Report it upstream rather than carrying the shim.

### 2. Calibration runs through ONNX Runtime, which rejects the graph at *load*

```txt
Fatal error: trt.plugins:AddScalar(-1) is not a registered function/op
```

This reads like a plugin-registration problem. It is not. ORT validates the graph schema when
**loading** the model, before any execution provider is consulted, and it will not load an op it has
no schema for. `calibration_eps=["trt"]` does not help — ORT still has to load the model to hand it
to the TensorRT EP. (This machine *does* have `TensorrtExecutionProvider`; the failure is earlier
than provider selection.)

## The workaround: calibrate a stand-in, transplant the ranges

A plugin almost always has a slower, ONNX-expressible equivalent — usually that is how it was
validated in the first place. `AddScalar(x)` is `Add(x, scalar)`. So:

1. Build a **reference graph** with the custom node replaced by standard ops.
2. Quantize *that* with ordinary PTQ. ORT runs it happily, and the ranges it measures are the real
   graph's ranges **because the two compute the same function** — a property the example checks
   numerically rather than assumes.
3. Transplant the plugin node back into the quantized graph, keeping the Q/DQ around it.

Result: 4 Q/DQ pairs inserted, plugin restored, and the engine builds.

### One more trap: `high_precision_dtype`

The default converts un-quantized tensors to FP16, wrapping the graph in `Cast` nodes. The plugin is
then asked for an I/O format combination it does not implement, and the build fails with:

```txt
Failed to find any supported plugin/custom tactic format
Could not find any implementation for node {ForeignNode[x_cast_to_fp16...y_cast_to_fp32]}
```

The message blames **tactics**; the cause is **dtypes**. `high_precision_dtype="fp32"` keeps the
surrounding graph in the format the plugin was written for, and the build succeeds. A plugin that
implements FP16 would not hit this — which is the general rule: quantizing around a plugin only works
to the extent the plugin's `supportsFormatCombination` covers what the surrounding graph became.

## Related

+ [`../BasicExample/`](../BasicExample/README.md) — the plugin used here.
+ [`../../02-API/Layer/QDQStructure/`](../../02-API/Layer/QDQStructure/README.md) — what Q/DQ placement
  means at the network level.
+ [`../../06-DLFrameworkTRT/ModelOptimizer/`](../../06-DLFrameworkTRT/ModelOptimizer/README.md) — the
  same toolkit on a plain PyTorch model, with no plugin in the way.
