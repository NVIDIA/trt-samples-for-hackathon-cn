# NonZero layer

+ NonZero layer. Computes the indices of the input tensor where the value is non-zero; the returned indices are in row-major order. This is a Data-Dependent-Shape (DDS) layer. See `case_simple` and `case_deprecated` in `main.py`.

+ Steps to run.

```bash
python3 main.py
```

+ Input / output data types.

| Item   | Data Type                                                       |
| :----- | :------------------------------------------------------------- |
| input  | `bool`, `int32`, `int64`, `float16`, `float32`, `bfloat16`      |
| output | `int32`, `int64` (selected by the `indices_type` attribute)    |

+ Attributes.

| Name           | Description                                        | Default              | Range                                    |
| :------------- | :------------------------------------------------ | :------------------- | :--------------------------------------- |
| `indices_type` | Data type of the output indices tensor.           | `trt.DataType.INT32` | `trt.DataType.INT32`, `trt.DataType.INT64` |

+ Shape / volume constraints.

  + Output shape is `[D, C]`, where `D = rank(input)` and `C` is the (data-dependent) count of non-zero elements.
  + Both input and output tensors support up to `2^31 - 1` elements.

+ Output Allocator MUST be used for Data-Dependent-Shape mode like this example.
  + In TensorRT-8.6 ~ TensorRT-9.X, `context.get_tensor_shape()` returns real output shape after the enqueue in DDS mode, so we can use it to get real output shape.
  + But this API always return shape with -1 in DDS mode since TensorRT-10, so we must use Output Allocator to get the real output shape.
