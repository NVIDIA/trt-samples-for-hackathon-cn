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
