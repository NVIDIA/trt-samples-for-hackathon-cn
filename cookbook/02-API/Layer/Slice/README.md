# Slice layer

+ Slice layer.

+ Steps to run.

```bash
python3 main.py
```

+ Extract a sub-tensor from the input tensor using `start`, `size` and `stride`, optionally padding out-of-bound coordinates according to `mode`. Refer to `case_simple` for the basic usage, `case_pad` for padding with `FILL` mode, `case_set_input` / `case_shape_input` for feeding parameters from other tensors.

+ Attributes.

| Attribute | Description | Default | Range |
| :-------: | :---------- | :-----: | :---: |
| start | First coordinate to use, one value per axis. | - | - |
| shape | Output dimensions (size), one value per axis. | - | - |
| stride | Stride between coordinates, one value per axis. | - | - |
| mode | How out-of-bounds coordinates are handled, see `trt.SampleMode`. | STRICT_BOUNDS | - |
| axes | Which axes `start`, `shape` and `stride` apply to (only when set via `set_input(5, ...)`). | all axes | - |

+ Available values of `trt.SampleMode`.

|     Name      |                             Comment                             |
| :-----------: | :------------------------------------------------------------: |
| STRICT_BOUNDS | $output[i] = data[i]$ if $0 \le i < w$, otherwise raise error. |
|     WRAP      |             $output[i] = data[\mathrm{mod}(i, w)]$             |
|     CLAMP     |    $output[i] = data[\min(\max(i, 0), w-1)]$ (clamp to edge)   |
|     FILL      | $output[i] = data[i]$ if $0 \le i < w$, otherwise fill value. |
|    REFLECT    | $output[i] = data[(w-1-i\%w)*(i/w\%2)+(i\%w)*(1-i/w\%2)]$      |

+ Input / output data types and shapes:
  + Input tensor (`T`) and output tensor share the same data type: `T` in [bool, int4, int8, int32, int64, float8, float16, float32, bfloat16].
  + `start`, `shape`, `stride`, `axes` tensors (when provided via `set_input`) are int32 or int64.
  + The optional fill-value tensor (input index 4, used with `FILL` mode) has data type `T`.
  + For input shape `[d0,...,dn-1]`, output shape is `[size0,...,sizen-1]` determined by `shape`.
