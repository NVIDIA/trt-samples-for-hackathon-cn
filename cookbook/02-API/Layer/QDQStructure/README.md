# QDQ structure

+ Quantize-Dequantize (QDQ) structure, built from a `Quantize` layer followed by a `Dequantize` layer.

+ Steps to run.

```bash
python3 main.py
```

+ The Quantize layer converts a floating-point tensor into a low-precision (e.g. INT8) tensor, and the Dequantize layer converts it back to floating-point. Refer to `case_simple` for the basic QDQ pair, `case_axis` for per-channel quantization, `case_set_input_zero_point` for supplying scale/zero-point through `set_input`, and `case_three_argument` for the strongly-typed three-argument form that sets the output data type directly.

+ Computation.

$$
\begin{aligned}
Quantize:   \ output &= \textbf{clamp}\left(\textbf{round}\left( \frac{input}{scale}\right ) + zeroPt \right) \\
Dequantize: \ output &= \left(input − zeroPt\right) \cdot scale
\end{aligned}
$$

+ Input / output tensors.

| Tensor     | Layer      | Role                | Data Type                                   | Notes                                                            |
| ---------- | ---------- | ------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| input      | Quantize   | Input               | `float16`, `bfloat16`, `float32` (`T1`)     |                                                                 |
| scale      | Quantize   | Input               | `T1`                                        | Build-time constant; scalar (per-tensor), 1-D (per-channel), or block-rank |
| zero_point | Quantize   | Input (optional)    | `float32` (`T2`)                            | Must contain only zeros; same shape as `scale`                  |
| output     | Quantize   | Output              | `int4`, `int8`, `float4`, `float8` (`T3`)   |                                                                 |
| input      | Dequantize | Input               | `int4`, `int8`, `float4`, `float8` (`T1`)   |                                                                 |
| scale      | Dequantize | Input               | `float16`, `bfloat16`, `float32` (`T3`)     | Build-time constant; same layout as above                       |
| zero_point | Dequantize | Input (optional)    | `float32` (`T2`)                            | Must match `scale` shape                                        |
| output     | Dequantize | Output              | `float16`, `bfloat16`, `float32` (`T3`)     |                                                                 |

+ Shape: input and output share shape `[a0, ..., an]`; each `scale` dimension equals the input dimension divided by the corresponding block size (1 for per-tensor / per-channel).

+ Attributes.

| Attribute   | Layer                 | Description                                                                                                     | Default   | Range / Valid Values                 |
| ----------- | --------------------- | ------------------------------------------------------------------------------------------------------------- | --------- | ------------------------------------ |
| axis        | Quantize / Dequantize | Quantization axis. For per-channel quantization it must be set explicitly; a negative axis raises an error.    | -1        | Valid axis index of the input tensor |
| to_type     | Quantize              | Data type of the quantized output tensor.                                                                      | `int8`    | `int4`, `int8`, `float4`, `float8`   |
| to_type     | Dequantize            | Data type of the dequantized output tensor.                                                                    | `float32` | `float16`, `bfloat16`, `float32`     |
| block_shape | Quantize / Dequantize | Shape of the quantization block; must match the rank of the input tensor. `-1` denotes a dimension fully blocked (block size equals the input extent on that dimension). | N/A | Same rank as input |

+ Notes.

+ The quantization axis must be specified for per-channel quantization, otherwise an error is raised:

```
[TensorRT] ERROR: 2: [scaleNode.cpp::getChannelAxis::20] Error Code 2: Internal Error ((Unnamed Layer* 2) [Quantize]: unexpected negative axis)
```

+ The `case_three_argument` case uses a strongly-typed network (`trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED`), where the output data type is passed directly to `add_quantize` / `add_dequantize` instead of relying on `BuilderFlag.INT8`.

## Block quantization (`block_quantization.py`)

`main.py` covers per-tensor and per-channel Q/DQ. **Block** quantization is the third granularity —
the one the MX formats are built on. The scale tensor has the same rank as the data, and
`block_shape` says how many elements share each scale:

| granularity | selected by | scales for a `[64, 32]` weight |
| --- | --- | ---: |
| per-tensor | – | 1 |
| per-channel | `axis` | 32 |
| block `[32, 1]` | `block_shape` | 64 |
| block `[16, 1]` | `block_shape` | 128 |

The scale shape is **derived, not free**: `data.shape[i] / block_shape[i]`. Measured on B200,
TensorRT 11.1.0.106:

### FP4 output is rejected

```txt
Blockwise quantization requires output type to be int8 or fp8e4m3
```

`IQuantizeLayer` block quantization accepts INT8 and FP8-E4M3 only. FP4 block quantization does
exist — through `IDynamicQuantizeLayer`, which [`../DynamicQuantize/`](../DynamicQuantize/README.md)
already covers.

### An E8M0 scale does not build

E8M0 is the shared-exponent scale type of the MX formats, and `trt.DataType.E8M0` exists — but
feeding one to `IQuantizeLayer` fails inside Myelin:

| scale type | standalone Q/DQ | feeding a MatMul |
| --- | --- | --- |
| float16 | builds | builds |
| **E8M0** | **fails** | **fails** |

```txt
MyelinCheckException: nvrtc_compile.cpp:1122: CHECK(success) failed. NVRTC Compilation failure
Could not find any implementation for node {ForeignNode[...Quantize...Dequantize]}
```

An NVRTC compilation error, not a usage message — nothing in it points at the scale type. An
earlier version of this example blamed the *missing consumer* instead, because the first failing
case happened to have both. The 2x2 above is what separated the two variables, and the consumer
turns out to be irrelevant.
