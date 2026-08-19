# Dumping a TensorRT network as an ONNX-like file

+ `OnnxLikeFromNetwork` walks a parsed `INetworkDefinition` and writes an ONNX
  file whose nodes are TensorRT layer types, so the network can be opened in
  Netron.

+ Steps to run.

```bash
python3 main.py
```

## What the parser actually produced

The reason to dump the network rather than read the source ONNX: they are not
the same graph. 12 ONNX nodes become 27 TensorRT layers.

```
source ONNX      : 12 nodes, 9 initializers
  {'Conv': 2, 'Relu': 3, 'MaxPool': 2, 'Reshape': 1, 'Gemm': 2, 'Softmax': 1, 'ArgMax': 1}
TensorRT network : 27 nodes, 0 initializers
  {'CONVOLUTION': 2, 'ACTIVATION': 3, 'POOLING': 2, 'SHUFFLE': 4, 'CONSTANT': 5,
   'MATRIX_MULTIPLY': 2, 'ELEMENTWISE': 2, 'SOFTMAX': 1, 'SHAPE': 1, 'CAST': 3,
   'TOPK': 1, 'SQUEEZE': 1}
```

The shuffles, casts and constants are all parser insertions — one `Gemm` becomes
`MATRIX_MULTIPLY` + `ELEMENTWISE`, `ArgMax` becomes `TOPK` + `SQUEEZE`, and
`Reshape` drags in a `SHAPE`/`CAST` chain. None of them are in the file that was
parsed. This is the view to reach for when a layer name in a build log or a
profiling report does not correspond to anything in the ONNX.

## It is not valid ONNX

Stated in the docstring (`[HIGHLY EXPERIMENTAL] ... not valid ONNX`) and worth
seeing, because the file opens fine in Netron and looks entirely legitimate:

```
onnx.checker: ValidationError: No Op registered for CONVOLUTION with domain_version of 11
```

The size is the second surprise. Zero initializers looks like the weights were
dropped — they were not. Every layer parameter becomes an ONNX **attribute**,
weights included:

```
initializers: 0 -- but the weights are still there, as attributes:
  first CONVOLUTION: kernel=800 floats, bias=32 floats, plus stride/padding/dilation
so the dump is 15.6 MB against a 12.5 MB source model
```

Protobuf stores a `FLOATS` attribute list far less compactly than an
initializer's packed `raw_data`, so the dump comes out *bigger* than the model it
was made from. Worth knowing before running this on a multi-GB network.

## Against the cookbook's own exporter

[`08-Advance/Subgraph/`](../../../../08-Advance/Subgraph/README.md) does the same
job with `tensorrt_cookbook.export_network_as_onnx`. Both were pointed at the
same network — an ONNX whose main graph holds a `Loop` whose body holds an `If`:

```
source ONNX: 8 nodes, 1 of them hold a sub-graph
polygraphy OnnxLikeFromNetwork : ValueError: Could not infer the attribute type from the elements of the passed Iterable
cookbook  export_network_as_onnx: 38 nodes written
```

Polygraphy's exporter cannot convert the loop and conditional boundary layers'
attributes and raises inside `gs.export_onnx`. The cookbook helper writes all 38
layers — TensorRT flattens both sub-graphs into a single layer list, with
`TRIP_LIMIT` / `RECURRENCE` / `CONDITION` / `CONDITIONAL_INPUT` /
`CONDITIONAL_OUTPUT` marking the boundaries.

So: `OnnxLikeFromNetwork` for a plain feed-forward network, the cookbook helper
when there is control flow. Neither produces a loadable model.
