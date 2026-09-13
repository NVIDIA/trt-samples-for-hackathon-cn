# Output Allocator

+ Write an `IOutputAllocator` for a detection head, where the number of output boxes is decided by the picture.

+ Steps to run.

```bash
python3 main.py
```

[`../../02-API/Layer/NonZero/`](../../02-API/Layer/NonZero/README.md) and
[`../../02-API/Layer/NMS/`](../../02-API/Layer/NMS/README.md) show that a layer *can* have a
data-dependent shape (DDS); `TRTWrapperDDS` hides the allocator so those examples stay about the
layer. This one is about the allocator itself, on the workload that forces you to write one.

The engine is a detection tail — NMS over 2048 anchors and 4 classes, then a Gather that turns the
selected indices into the surviving boxes:

```txt
boxes  [1, 2048, 4] ---+
                       +--> NMS --> selected_indices [n, 3]  (DDS)
scores [1, 2048, 4] ---+            num_output_boxes  []      (fixed)
                                        |
                                        +--> Gather --> selected_boxes [n, 4]  (DDS)
```

Measured on B200, TensorRT 11.1.0.106.

## The interface, and when it is called

Two methods, both called **during** `execute_async_v3`, which is the whole point — neither answer
exists before the data has been seen:

| Method | Contract |
| ------ | -------- |
| `reallocate_output_async(name, old_address, size, alignment, stream)` | return a device address holding at least `size` bytes |
| `notify_shape(name, shape)` | receive the shape TensorRT actually produced |

Attaching one is a single call: `context.set_output_allocator(name, allocator)`. Only DDS outputs
need it — `num_output_boxes` above has shape `()` and is bound normally with `set_tensor_address`.

## Grow-only, and what it costs

Three images with 3, 200 and 17 objects — deliberately grow, grow, shrink:

```txt
image with   3 objects -> selected_indices request 24576 B -> cudaMalloc
                          selected_boxes   request   512 B -> cudaMalloc
image with 200 objects -> selected_indices request 24576 B -> reused
                          selected_boxes   request  3584 B -> cudaMalloc
image with  17 objects -> selected_indices request 24576 B -> reused
                          selected_boxes   request   512 B -> reused
```

Never shrinking is the right default: a detector's box count jitters frame to frame, and freeing
plus re-allocating every frame puts a `cudaMalloc` on the critical path for nothing. Over the three
images that is **2 `cudaMalloc` calls for 6 requests**, and it converges to zero.

## Not all DDS outputs are equally dynamic

This is the finding that changes how you budget memory:

| Tensor | sizes requested across the three images |
| ------ | --------------------------------------- |
| `selected_indices` (straight off NMS) | `[24576]` — **always the upper bound** |
| `selected_boxes` (a Gather downstream) | `[512, 3584]` — the true size |

TensorRT asks for the NMS output's worst case every single time, no matter what the image holds.
Only the downstream tensor is requested at its real size. So an allocator saves memory on the second
kind and nothing at all on the first:

```txt
Worst-case pre-allocation :      57348 B (0.05 MiB)
Peak held by the allocator:      28160 B (0.027 MiB)
Saved                     :      29188 B, 2.0x smaller
```

2x, not 100x. "DDS means you pay for what you use" is only half true, and which half you get depends
on where the tensor sits relative to the layer that made the shape data-dependent.

## Two traps when reading the result back

### `context.get_tensor_shape()` never resolves for a DDS output

After a completed inference it still returns `(-1, 4)`. A merely *dynamic* shape is resolved on the
context once the input shapes are set, and sizing the device-to-host copy with `get_tensor_shape` is
the normal idiom there — port that code to a DDS output and it computes a negative byte count.
`notify_shape` is the only channel that carries the answer.

### Capacity is not shape

The allocator holds the high-water mark of every image seen so far. After the 200-object image the
buffer has room for 224 rows; the next image has 5 boxes. Dividing `n_byte` by the item size reports
**224 boxes instead of 5**, silently returning stale boxes from the previous frame.

```txt
200 objects: notify_shape=(200, 4), context.get_tensor_shape=(-1, 4), buffer capacity would suggest 224 rows
  5 objects: notify_shape=(5, 4),   context.get_tensor_shape=(-1, 4), buffer capacity would suggest 224 rows
```

## Why not the upstream Faster R-CNN sample

`samples/python/dds_faster_rcnn` demonstrates the same API on a real Faster R-CNN. It needs a
~160 MB `FasterRCNN-12.onnx` download plus COCO labels and a visualisation stack, and since
2026-09-01 the cookbook's examples deliberately do not download anything at run time. The detection
*tail* above reproduces every API interaction — DDS from NMS, reallocation, `notify_shape`, the
worst-case comparison — in a self-contained file that builds in a second. Read the upstream sample
for the full pre/post-processing pipeline.

## Related

+ [`../../02-API/Layer/NMS/`](../../02-API/Layer/NMS/README.md),
  [`../../02-API/Layer/NonZero/`](../../02-API/Layer/NonZero/README.md) — the layers that create DDS.
+ [`../../05-Plugin/DataDependentShape/`](../../05-Plugin/DataDependentShape/README.md) — producing a
  data-dependent shape from inside a plugin.
+ `TRTWrapperDDS` and `CookbookOutputAllocator` in `tensorrt_cookbook/utils_class.py` — the
  ready-made versions the other examples use.
