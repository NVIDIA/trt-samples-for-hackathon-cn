# Dynamic quantize layer

+ Dynamic quantize layer.

+ Steps to run.

```bash
python3 main.py
```

+ Quantize a floating-point tensor into a low-precision (FP4 / FP8) tensor while computing the per-block scale factors dynamically at run time (as an extra output). Refer to `case_v1` / `case_v2` for 1-D and 2-D block quantization, and `case_v1_double_quantization` / `case_v2_double_quantization` for double (two-level) quantization using the optional `double_quant_scale` input via `set_input(1, ...)`.

+ Input / output tensors.

| Tensor             | Role             | Data Type                               | Notes                                             |
| ------------------ | ---------------- | --------------------------------------- | ------------------------------------------------- |
| input              | Input            | `float16`, `bfloat16`, `float32` (`T1`) |                                                   |
| double_quant_scale | Input (optional) | `T1` or `float32`                       | Scalar, build-time constant (double quantization) |
| output (data)      | Output 0         | `float4`, `float8` (`T2`)               | Quantized tensor, same shape as input             |
| output (scale)     | Output 1         | `float8`, `float32` (`T3`)              | Per-block scale factors                           |

+ Shape: input and output-data share shape; the output-scale has the same rank as the input, with each quantization dimension divided by the corresponding block size (e.g. for input `(D0, D1, D2)` with `axis=2`, output-scale is `(D0, D1, D2 / block_size)`). Input/output are 2-D or 3-D for 1-D block quantization and 2-D, 3-D, or 4-D for 2-D block quantization.

+ Attributes.

| Attribute             | Description                                                                                                                                                          | Default              | Range / Valid Values                         |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | -------------------------------------------- |
| axis                  | (1-D block, `add_dynamic_quantize`) The axis sliced into blocks. Must be the last or second-to-last dimension.                                                       | Constructor argument | Last or second-to-last dimension             |
| block_size            | (1-D block) Number of elements sharing a scale factor.                                                                                                               | Constructor argument | 16 (typical for FP4) or 32 (typical for FP8) |
| block_shape           | (2-D block, `add_dynamic_quantize_v2`) The shape of the block to be quantized. Compile-time constant with the same rank as the input; `-1` matches the input extent. | Constructor argument | Same rank as input                           |
| to_type (output_type) | Data type of the quantized output.                                                                                                                                   | Constructor argument | `DataType.FP4`, `DataType.FP8`               |
| scale_type            | Data type of the scale factor.                                                                                                                                       | Constructor argument | `DataType.FP8`, `DataType.FLOAT`             |
