# Empty Tensor

+ Zero-volume tensors, in the situations where they actually occur.

A detector whose score threshold keeps **no** box on this frame, and a serving stack that hands the
model a batch of zero rows.

+ Steps to run.

```bash
python3 main.py
```

TensorRT itself handles empty tensors. The example is about the three ways the program *around*
them goes wrong — all measured here on TensorRT 11.1.0.106.

## The cases

| Case | What it shows |
| ---- | ------------- |
| `case_no_detection` | `Greater` → `NonZero` → `Gather`, the tail of any detector. Nothing passes the threshold → `kept_box` comes back with shape `(0, 4)`; the same engine and the same call return `(3, 4)` when three boxes pass. **An empty result is a normal result** — no host-side "if empty then skip" branch is needed |
| `case_empty_tensor_needs_a_valid_address` | `cudaMalloc(0)` succeeds and returns **address 0**. Binding NULL to an input makes `enqueueV3` return `False` and run nothing, and the output buffer keeps whatever it held before |
| `case_reduce_over_empty_axis` | Reducing over 0 elements: `SUM = 0`, but **`MAX = -inf` and `AVG = NaN`** |
| `case_profile_must_cover_zero` | A shape of `0` is only legal if the optimization profile allows it. `set_input_shape` returns `False` and leaves the previous shape in place |

## The trap that is worth the example

```txt
cudaMalloc(0) -> status=cudaSuccess, address=0        (success, and the address is NULL)
empty input bound to NULL (cudaMalloc(0)): enqueueV3=False, merged=[-999. -999. -999. -999.]
empty input bound to a 1-byte allocation: enqueueV3=True,  merged=[   0.    1.    2.    3.]
```

The failing run leaves a **plausible-looking output buffer** behind. In the first version of this
very example that mistake silently corrupted the third case: with the empty input bound to NULL the
reduction never ran, the output buffer was never written, and `SUM/MAX/AVG` all read back as `0` —
so the example almost shipped the conclusion "MAX over nothing is 0". Bound correctly, the same
network answers:

```txt
empty : sum=[0. 0. 0.], max=[-inf -inf -inf], avg=[nan nan nan]
4 rows: sum=[18. 22. 26.], max=[ 9. 10. 11.], avg=[4.5 5.5 6.5]
```

`-inf` and `NaN` are the honest identities of MAX and AVG over an empty set, and both travel: one
`NaN` average poisons every later average, comparison and loss it touches. Branch on the count
*before* the reduction; afterwards the value is indistinguishable from a genuine `NaN`.

Two rules follow, and both are about return values nobody reads:

+ **Allocate at least one byte for a zero-volume tensor** (`cudaMalloc(max(n_byte, 1))`), and check
  what `enqueueV3` returns. This example is what made `tensorrt_cookbook/utils_class.py` do the
  former — all four of its buffer allocators used to call `cudaMalloc(n_byte)` and hand TensorRT a
  NULL address whenever a tensor was empty.
+ **A model that may see an empty batch needs `min = 0` in its profile.** With `min = 1`,
  `set_input_shape([0, 2])` is rejected (`does not satisfy any optimization profiles. Valid range
  for profile 0: [1,2]..[8,2]`) and returns `False`; a caller that ignores the return value runs
  the previous shape's data again.

## Related

+ [`../../02-API/Layer/NonZero/`](../../02-API/Layer/NonZero/README.md) — the layer that produces
  the data-dependent shape used here.
+ [`../../04-Feature/OutputAllocator/`](../../04-Feature/OutputAllocator/README.md) — how a
  data-dependent output shape is received at run time.
+ [`../MultiOptimizationProfile/`](../MultiOptimizationProfile/README.md) — profile ranges in
  general.
