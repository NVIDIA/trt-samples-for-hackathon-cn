# Device memory

+ Keep tensors on the GPU across a `TrtRunner` call, with `polygraphy.cuda`.

+ Steps to run.

```bash
python3 main.py
```

`polygraphy.cuda` is three small classes:

| class | owns memory? | interface |
| --- | --- | --- |
| `DeviceArray(shape, dtype)` | yes | `copy_from` `copy_to` `numpy` `view` `resize` `free` `raw` `ptr` `shape` `dtype` `nbytes` `allocated_nbytes` |
| `DeviceView(ptr, shape, dtype)` | **no** | `copy_to` `numpy` `ptr` `shape` `dtype` `nbytes` |
| `Stream()` | — | `synchronize` `free` `ptr` |

`array.view()` returns the same pointer, so it allocates nothing and copies
nothing:

```
array : DeviceArray[(dtype=float32, shape=(2, 3)), ptr=0x71d376600000]
view  : DeviceView[(dtype=float32, shape=(2, 3)), ptr=0x71d376600000]
view.ptr == array.ptr : True  (view() is not a copy)
```

Two smaller notes from the same case. `resize()` never shrinks the allocation —
after `resize((1, 3))` the array reports `nbytes 12` while still holding 24 B.
And reading `.dtype` is itself deprecated:

```
view.dtype returns dtype('float32') and warns: Using NumPy data types in
DeviceView/DeviceArray is deprecated and will be removed in Polygraphy 0.55.0.
```

It returns a NumPy dtype today and a Polygraphy `DataType` later, so
`np.empty(..., dtype=view.dtype)` is code with an expiry date on it.

This example also closes the loose end from
[`../10-PyTorchTensors/`](../10-PyTorchTensors/README.md): the object with no
`.device` and no `.cpu()` was a `DeviceView`, and the table above is all of it.

## The runner really does use your pointer

`context.get_tensor_address` says which buffer TensorRT was bound to:

```
our DeviceArray lives at        : 0x4578000000
bound address after NumPy feed  : 0x4578020000  -> runner's own buffer
bound address after view feed   : 0x4578000000  -> ours (True)
outputs identical               : True
```

A NumPy feed makes `TrtRunner` allocate a device buffer of its own and copy into
it. A `DeviceView` feed binds your address directly. A `DeviceArray` works as a
feed too — it is a `DeviceView` subclass.

## Outputs left on the device are the runner's buffer, not yours

`copy_outputs_to_host=False` returns `DeviceView`s pointing into the runner's
output allocator, and that allocation is reused on the next inference:

```
output type: DeviceView, has .cpu(): False, has .device: False
output address is the same buffer both times: True
the reference we kept now reads              : [-0.004, -0.015, 0.032, 0.057] ...
...which is the *second* result              : True
...and no longer the first                   : False
```

This is the device-side twin of the host-buffer reuse in
[`../04-ExtendInterop/`](../04-ExtendInterop/README.md), and it is harder to
notice, because a `DeviceView` has no contents to look at. Copy out with
`numpy()` or `copy_to(buffer)` before the next `infer`.

The payoff is chaining: one runner's output view goes straight into the next
runner's feed dict with no host memory in between.

## What the round trip actually costs

Four combinations of (input on host / on device) × (output to host / left on
device), on two models:

```
MNIST  3 KB (0.003 MB in)
  host in, host out  :   0.582 ms
  host in, device out:   0.458 ms
  device in, host out:   0.510 ms
  device in, dev out :   0.397 ms
  staying on the device is  1.46x, saving 0.185 ms -- 32% of the round trip was copying

synthetic 25 MB (25.166 MB in)
  host in, host out  :   9.328 ms
  host in, device out:   2.572 ms
  device in, host out:   7.358 ms
  device in, dev out :   0.290 ms
  staying on the device is 32.14x, saving 9.038 ms -- 97% of the round trip was copying
```

On MNIST almost none of the 0.185 ms is bandwidth — it is the fixed cost of
issuing two copies and synchronizing. On the 25 MB tensor (`y = x + x`, one add
per element) everything except 0.290 ms is PCIe.

So the technique scales with bytes. Below roughly a megabyte the payoff is a
fixed fraction of a millisecond, which is not obviously worth the two hazards
below.

## A view does not keep the memory alive

A `DeviceView` is an integer. Both ways of losing the memory under it are things
the owner does to itself, and neither is visible from the view.

`free()`, then the next allocation lands on the same address:

```
view of a live array at 0x457801f200 reads [1.0, 1.0, 1.0, 1.0]
after free(): owner ptr is 0, the view still says 0x457801f200
a fresh allocation landed at 0x457801f200 -- same address: True
the stale view now reads [9.0, 9.0, 9.0, 9.0] and reports no error at all
```

`resize()` to something larger — `resize` frees and re-mallocs internally, so
views taken before it are dangling too. When the new allocation lands elsewhere,
reading the view is not an exception:

```
reading a view across resize(), in a child process: returncode -11 (-11 is SIGSEGV)
stdout '', python-level traceback: False
```

(run in a child process for obvious reasons)

The rule has to be structural: a view must not outlive its `DeviceArray`. Note
this cuts against the context-manager habit — `with DeviceArray(...) as a:` frees
on exit, so any view that escapes the `with` block is already dangling.

## The stream that is not asynchronous

`copy_from(buffer, stream)` calls `cudaMemcpyAsync`, but on ordinary pageable
host memory that has to stage through a driver buffer and blocks anyway:

```
H2D 256 MB, pageable (numpy): copy_from returned after 22.593 ms, synchronized at 22.625 ms
H2D 256 MB, pinned   (torch): copy_from returned after  0.102 ms, synchronized at 12.230 ms
```

Which means a missing `synchronize()` is written but not observable — with a
NumPy buffer the data is always all there:

```
D2H pageable (numpy): 100.0% of the data was there before synchronize(), 100.0% after
D2H pinned   (torch):   2.5% of the data was there before synchronize(), 100.0% after
```

Pinning the host buffer is the actual win (0.102 ms to issue instead of 22.6 ms
of blocking), and it is also what makes the pre-existing bug start firing. The
exact percentage is a race and moves between runs; "not all of it" does not.

Pass a `Stream` and you own the `synchronize()`, whether or not today's buffer
type makes forgetting it survivable.
