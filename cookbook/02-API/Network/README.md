# Network

+ Steps to run.

```bash
python3 main.py               # INetworkDefinition attributes
python3 weight_transport.py   # move a trained state_dict into a hand-built network, without ONNX
```

## `weight_transport.py` — the `.wts` convention

+ The question this answers: *I have a PyTorch `state_dict` and I want to call `add_convolution_nd`
  myself. How do the weights get from one to the other?* ONNX is the usual answer; this is what the
  other path looks like, using the `.wts` format from
  [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx).

### The format

```
<blob-count>
<name> <n-value> <hex0> <hex1> ...
...
```

Each float32 is `struct.pack(">f", value).hex()` — big-endian IEEE-754, eight ASCII characters. The
file is pure text, so there is no byte order to get wrong when it moves between machines, and it can
be diffed, grepped and pasted into a bug report.

| | |
| --- | --- |
| Round trip | **bit-exact** for every blob — hex float32 is lossless, not "close enough" |
| Size | 81,870 bytes for 9,088 values = **2.25x** the raw 36,352 bytes, and **2.2x** an `.npz` |
| Load | **2x** slower than `np.load` |
| Shapes | **not stored.** Only a flat count, so the consumer must know every blob's layout |

For anything large this is a bad trade — use `.npz` or safetensors. What survives is the *idea*:
ship named weights beside the plan so a hand-built network can look them up.

### `register_buffer` — exporting things that are not parameters

`state_dict()` serializes buffers as well as parameters, so anything registered as a buffer is
exported for free. This is how `yolov5/gen_wts.py` ships its anchor grid: a tensor *derived* from
other tensors at export time that the TensorRT network would rather not recompute. The example
registers a per-channel norm the same way and reads it back out of the file.

### Two things that bite, both silent

**1. The default builder config enables TF32.** Same weights, same graph, one flag:

| | max &#124;TensorRT − torch&#124; |
| --- | --- |
| default (TF32 on) | 2.643e-04 |
| `clear_flag(trt.BuilderFlag.TF32)` | 8.941e-08 |

A **2956x** difference, and nothing is wrong with the weights. TF32 rounds the mantissa to 10 bits
inside the MatMul. This is the first thing to check when a hand-built network "does not match", and
it is very easy to misread as a transposed blob or a bad reshape.

**2. `trt.Weights` wraps a pointer — it does not copy and does not keep the object alive.** The
values are read when the engine is *built*, not when `add_convolution_nd` is called. The example
hands TensorRT a bias array, overwrites it with `999.0` before `build()`, and the engine computes
`999.0`. Letting the array be garbage collected instead is a use-after-free with no error message.
Every hand-built network needs a keep-alive list.

### Also worth knowing

`.wts` stores no shapes and `nn.Linear` stores its weight as `(out_features, in_features)`, so the
matmul has to transpose the second operand. Both are supplied by hand here, and both are silent if
you get them wrong in a way that still has the right element count.
