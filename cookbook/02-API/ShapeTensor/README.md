# Shape Tensor

+ The input whose **values** are shapes: how it differs from an execution tensor, and the four ways to get it wrong.

+ Steps to run.

```bash
python3 main.py
```

TensorRT has two kinds of input and almost every rule differs between them:

| | execution tensor | **shape tensor** |
| --- | --- | --- |
| what varies at runtime | its shape | its **values** |
| lives on | device | **host** |
| profile method | `set_shape` | **`set_shape_input`** |
| context method | `set_input_shape` | **`set_tensor_address`** (host pointer) |
| identified by | – | `engine.is_shape_inference_io(name)` |

The names are the trap. `set_shape` / `set_input_shape` name the *shape* of a tensor;
`set_shape_input` names the *values* of one. Measured on B200, TensorRT 11.1.0.106.

## Identifying one

```txt
name      mode    dtype           build shape     is_shape_inference_io
data      INPUT   FLOAT           (-1, -1, -1)    False
newShape  INPUT   INT32           (3,)            True
output    OUTPUT  FLOAT           (-1, -1, -1)    False
```

Neither dtype nor rank identifies a shape tensor — an INT32 rank-1 input is usually an ordinary
execution tensor. `is_shape_inference_io` is the only reliable test. Note also that
`ITensor.is_shape_tensor` is **False** at network-definition time for an input that later becomes
one: a tensor becomes a shape tensor by being *consumed* as a shape, so the property is not yet true
when you create the input.

## INT32 and INT64 both work

Both widths build and run; what has to match is the host buffer you hand over. A mismatched width
does not raise — it reads neighbouring bytes.

## 0-D shape tensors

A scalar whose value steers the graph. `IFillLayer` in `LINSPACE` mode is the smallest real use:
`alpha` (start) is rank 0, and `delta` must be rank **1**, one per output dimension. Getting that
backwards fails at build with `requires that input at index 2 have rank 1`.

```txt
start=5    -> linspace [5, 6, 7, 8, 9, 10, 11, 12]
start=50   -> linspace [50, 51, 52, 53, 54, 55, 56, 57]
```

Bind it as `np.array(v)`, not `np.array([v])` — the second is rank 1.

## A device pointer is fatal, not an error

`set_tensor_address` takes an integer, so a device pointer is accepted without complaint and then
**dereferenced on the host** during shape inference. The result is not an exception and not a wrong
answer:

```txt
child exit code: -11 (SIGSEGV)
Signal code: Invalid permissions (2)
Failing at address: 0x78b3dd200800
libnvinfer.so.11(+0x1e9ce76)
```

The stack says nothing about shape tensors. The example runs this deliberately-wrong binding in a
**child process**, because it is the only way to demonstrate it without taking the run down — the
child prints `bound without complaint` and then dies.

## The wrong API does not complain either

```txt
set_input_shape('newShape', (3,))            -> accepted (it names the shape, not the values)
profile.set_shape('newShape', [1],[3],[3])   -> accepted (the profile has no engine to check against)
```

Both mistakes are silent at the point they are made. The failure surfaces later as an unspecified
binding or a wrong output shape, a long way from the line that caused it.

## `infer_shapes` is the cheapest debugger

It returns the names that are still unresolved, so it answers "what have I forgotten to bind":

```txt
nothing set yet          : ['data', 'newShape']
execution tensor set     : ['newShape']
shape tensor set as well : []            <- empty list = everything resolved
```

Call it before `execute_async_v3` and the two silent mistakes above become a printed list.

## Related

+ [`../ISymExprs/`](../ISymExprs/README.md) — the symbolic-expression machinery behind shape inference.
+ [`../../05-Plugin/ShapeInputTensor/`](../../05-Plugin/ShapeInputTensor/README.md) — consuming a shape
  input inside a plugin.
+ [`../../05-Plugin/DataDependentShape/`](../../05-Plugin/DataDependentShape/README.md) and
  [`../../04-Feature/OutputAllocator/`](../../04-Feature/OutputAllocator/README.md) — the other
  direction, where the *output* shape is not known until the data is seen.
+ [`../Layer/Fill/`](../Layer/Fill/README.md), [`../Layer/Slice/`](../Layer/Slice/README.md),
  [`../Layer/Resize/`](../Layer/Resize/README.md) — layers that take shape inputs.
