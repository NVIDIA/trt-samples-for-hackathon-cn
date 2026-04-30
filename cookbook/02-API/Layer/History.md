# History

## Overview

This file documents the refactoring of TensorRT Layer example code and documentation under `cookbook/02-API/Layer/`. The refactoring standardized README.md headers, added Attributes tables sourced from NVIDIA documentation, added I/O constraint comments in `main.py` files, moved `check_api_coverage()` calls to the correct position (last statement before `tw.build()`/`tw.setup()`), and added inline `[Optional]` comments with default values to layer attribute assignments.

---

## Directories with Multiple Python Scripts

**NONE.** All Layer subdirectories have exactly one `main.py` (or no Python file if they are reference-only).

---

## Parameter Defaults Not Found in Docs

The following layer attributes could not have their default values confirmed from the NVIDIA TensorRT documentation. For each, an experimental snippet is provided to extract the default at runtime.

### `alpha` and `beta` (ActivationLayer)

NVIDIA docs do not specify numeric defaults for `alpha` and `beta`; existing code says `0` but this is unconfirmed.

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
inp = network.add_input("x", trt.DataType.FLOAT, (1,))
layer = network.add_activation(inp, trt.ActivationType.RELU)
print(layer.alpha)  # Default alpha
print(layer.beta)   # Default beta
```

### `dtype` (CastLayer)

Default is the `toType` constructor argument, not a fixed API default.

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
inp = network.add_input("x", trt.DataType.FLOAT, (1,))
layer = network.add_cast(inp, trt.DataType.BOOL)
print(layer.dtype)  # equals trt.DataType.BOOL
```

### `num_output_maps`, `kernel_size_nd`, `kernel`, `bias` (ConvolutionLayer / DeconvolutionLayer)

These are required constructor arguments repurposed as optional setters; no standalone default exists.

```python
import tensorrt as trt
import numpy as np
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
inp = network.add_input("x", trt.DataType.FLOAT, (1, 4, 8, 8))
w = trt.Weights(np.zeros((8, 4, 3, 3), dtype=np.float32))
layer = network.add_convolution_nd(inp, 8, (3, 3), w)
print(layer.num_output_maps)  # set by constructor
print(layer.kernel_size_nd)
```

### `collective_operation`, `reduce_op`, `root` (DistCollectiveLayer)

Constructor-required parameters; no defaults defined in docs.

```python
import tensorrt as trt
# These are required args — no default can be printed without a full
# multi-GPU setup. Inspect the layer after construction to observe values.
```

### `axis`, `block_size`, `block_shape`, `to_type`, `scale_type` (DynamicQuantizeLayer)

All are required constructor arguments for `add_dynamic_quantize` / `add_dynamic_quantize_v2`.

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
inp = network.add_input("x", trt.DataType.FLOAT, (4, 8))
layer = network.add_dynamic_quantize(inp, 1, 4, trt.DataType.INT8, trt.DataType.FLOAT)
print(layer.axis)
print(layer.block_size)
print(layer.to_type)
print(layer.scale_type)
```

### `op` (ElementWiseLayer)

TRT docs page returned 404; default is the op passed at construction time.

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
inp = network.add_input("x", trt.DataType.FLOAT, (1,))
inp2 = network.add_input("y", trt.DataType.FLOAT, (1,))
layer = network.add_elementwise(inp, inp2, trt.ElementWiseOperation.SUM)
print(layer.op)  # trt.ElementWiseOperation.SUM (set by constructor)
```

### `name` (IfConditional structure)

TRT docs page returned 404; auto-generated name format is unconfirmed.

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
if_structure = network.add_if_conditional()
print(if_structure.name)  # prints auto-generated name
```

### `window_size`, `alpha`, `beta`, `k` (LRNLayer)

All are set at construction time via `add_lrn()`; no fixed defaults in docs.

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
inp = network.add_input("x", trt.DataType.FLOAT, (1, 4, 8, 8))
layer = network.add_lrn(inp, 5, 1.0, 0.5, 2.0)
print(layer.window_size)
print(layer.alpha)
print(layer.beta)
print(layer.k)
```

### `shape`, `weights`, `op` (ConstantLayer / ElementWiseLayer inside LoopStructure)

Set at construction time; NVIDIA docs Loop page has no Attributes section.

```python
import tensorrt as trt
import numpy as np
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
w = trt.Weights(np.array([1.0], dtype=np.float32))
layer_t = network.add_constant((1,), w)
print(layer_t.shape)
print(layer_t.weights)
```

### `metadata`, `num_ranks` (MoELayer)

Default values not confirmed in NVIDIA docs; assumed `""` and `1` respectively.

```python
import tensorrt as trt
# Construct a minimal MoE layer, then inspect:
# print(layer.metadata)   # expected: ""
# print(layer.num_ranks)  # expected: 1
```

### `swigluParamLimit`, `swigluParamAlpha`, `swigluParamBeta`, `quantizationBlockShape`, `dynQOutputScaleType` (MoELayer)

Listed as "Not supported yet" in NVIDIA docs; no defaults available.

### `indices_type` (NMSLayer)

NVIDIA docs describe type T2 (int32 or int64) but do not specify a default for the property setter.

```python
import tensorrt as trt
# After constructing an NMS layer:
# print(layer.indices_type)  # expected: trt.DataType.INT32
```

### `axes` (NormalizationLayer)

Set by the `add_normalization_v2()` constructor argument; no separate documented default.

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
inp = network.add_input("x", trt.DataType.FLOAT, (1, 4, 8))
scale = network.add_input("s", trt.DataType.FLOAT, (4,))
bias = network.add_input("b", trt.DataType.FLOAT, (4,))
layer = network.add_normalization_v2(inp, scale, bias, 1 << 2)
print(layer.axes)
```

### `axis` (OneHotLayer)

Required positional argument to `add_one_hot()`; no standalone default documented.

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
inp = network.add_input("x", trt.DataType.INT32, (4,))
depth = network.add_input("d", trt.DataType.INT32, ())
vals = network.add_input("v", trt.DataType.FLOAT, (2,))
layer = network.add_one_hot(inp, vals, depth, 1)
print(layer.axis)
```

### `pre_padding_nd`, `post_padding_nd` (PaddingLayer)

Constructor arguments; no numeric defaults in NVIDIA docs.

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
inp = network.add_input("x", trt.DataType.FLOAT, (1, 4, 8, 8))
layer = network.add_padding_nd(inp, (1, 1), (1, 1))
print(layer.pre_padding_nd)
print(layer.post_padding_nd)
```

### `stride_nd` (PoolingLayer)

Docs say "same as window_size_nd" but do not state this as a confirmed default.

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
inp = network.add_input("x", trt.DataType.FLOAT, (1, 4, 8, 8))
layer = network.add_pooling_nd(inp, trt.PoolingType.MAX, (2, 2))
print(layer.stride_nd)  # expected: same as window_size = (2, 2)
```

### `axis` (Quantize/Dequantize)

NVIDIA docs infer default -1 from error behavior (must be set explicitly); not stated in docs.

```python
import tensorrt as trt
import numpy as np
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
inp = network.add_input("x", trt.DataType.FLOAT, (1, 4))
scale = network.add_input("s", trt.DataType.FLOAT, (1,))
layer_q = network.add_quantize(inp, scale, trt.DataType.INT8)
print(layer_q.axis)  # expected: -1
```

### `axes` (ReduceLayer), `shape` (ResizeLayer), `reshape_dims` (ShuffleLayer)

Set by constructor argument; no independent defaults documented.

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
inp = network.add_input("x", trt.DataType.FLOAT, (1, 4, 8))
# Reduce
layer_r = network.add_reduce(inp, trt.ReduceOperation.SUM, 1 << 1, False)
print(layer_r.axes)
# Resize
layer_res = network.add_resize(inp)
print(layer_res.shape)
# Shuffle
layer_sh = network.add_shuffle(inp)
print(layer_sh.reshape_dims)
```

### `start`, `shape`, `stride` (SliceLayer)

Set via constructor arg or `set_input()`; no documented defaults in NVIDIA docs.

```python
import tensorrt as trt
import numpy as np
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
inp = network.add_input("x", trt.DataType.FLOAT, (1, 4, 8))
layer = network.add_slice(inp, (0, 0, 0), (1, 2, 4), (1, 1, 1))
print(layer.start)
print(layer.shape)
print(layer.stride)
```
