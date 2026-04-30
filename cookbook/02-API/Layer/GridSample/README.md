# GridSample layer

+ GridSample layer.

+ Steps to run.

```bash
python3 main.py
```

+ Sample the input tensor at the (normalized) coordinates given by the grid tensor, see `case_simple`.

+ Input / output data type and shape:
  + Input tensor `input` of type `T` with shape `[N, C, H, W]` and grid tensor of type `T` with shape `[N, H_out, W_out, 2]` (last dimension holds normalized `(x, y)` coordinates in $[-1, 1]$).
  + Output tensor of type `T` with shape `[N, C, H_out, W_out]`.
  + `T` in [float16, float32, bfloat16]. Each of input, grid and output can have up to $2^{31}-1$ elements.

+ Available values of `trt.InterpolationMode`.

| Name    | Comment                            |
| :------ | :--------------------------------- |
| NEAREST | Nearest neighbor interpolation     |
| LINEAR  | Bilinear interpolation             |
| CUBIC   | Bicubic interpolation              |

+ Available values of `trt.SampleMode` (handling of grid coordinates that fall outside the input boundary).

| Name          | Comment                                                                                     |
| :------------ | :----------------------------------------------------------------------------------------- |
| STRICT_BOUNDS | Out-of-bound coordinates are illegal (behavior undefined / not allowed)                     |
| WRAP          | Out-of-bound coordinates wrap around to the opposite side of the input                       |
| CLAMP         | Out-of-bound coordinates are clamped to the nearest border value                            |
| FILL          | Out-of-bound coordinates are assigned the value 0                                           |
| REFLECT       | Out-of-bound coordinates are reflected across the input borders until they become in-bounds |

+ Attributes

| Name               | Description                                              | Type                  | Default                      | Range                                     |
| :----------------- | :------------------------------------------------------ | :-------------------- | :--------------------------- | :---------------------------------------- |
| align_corners      | Whether the corner pixels of input and grid are aligned | bool                  | False                        | {True, False}                             |
| interpolation_mode | Interpolation technique                                 | trt.InterpolationMode | trt.InterpolationMode.LINEAR | NEAREST, LINEAR, CUBIC                     |
| sample_mode        | Handling of out-of-bound grid locations                 | trt.SampleMode        | trt.SampleMode.FILL          | STRICT_BOUNDS, WRAP, CLAMP, FILL, REFLECT |
