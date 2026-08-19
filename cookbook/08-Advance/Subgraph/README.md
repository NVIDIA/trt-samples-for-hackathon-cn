# Subgraph

+ Use cases of parsing ONNX file with subgraph into TensorRT.

+ Steps to run.

```shell
python3 main.py
```

## What TensorRT does with a sub-graph

`torch.jit.script` keeps the data-dependent `for` and the `if`, so the exported
ONNX nests three levels deep:

```
main graph          8 nodes, one of them a `Loop`
  Loop.body        12 nodes, one of them an `If`
    If.then/else    2 nodes
```

`INetworkDefinition` has **no nesting**. The parser flattens everything into one
list of 38 layers and the control flow survives only as boundary layer types:

| ONNX | TensorRT layers |
| --- | --- |
| `Loop` | `TRIP_LIMIT` x2, `RECURRENCE` x3, `LOOP_OUTPUT` x1 (`ITERATOR` when the body indexes the scan input) |
| `If` | `CONDITION` x1, `CONDITIONAL_INPUT` x2, `CONDITIONAL_OUTPUT` x1 |

So `print_network` and `export_network_as_onnx` both show **one flat graph**:
`export_network_as_onnx` writes 38 nodes of which **0 hold a sub-graph**, and the
`Loop` node you could click into in Netron is gone. The file is ONNX-*like*, not
valid ONNX -- `TripLimit` / `Recurrence` / `ConditionalInput` are TensorRT layer
types, not ONNX operators.

### `layer.metadata` records provenance, but only for `If`

The one trace of where a layer came from is asymmetric:

```
29 [ONNX Layer: Add_23      | sub_graph1 (then_branch) | If_22 (If)]   <- branch named
30 [ONNX Layer: Identity_24 | sub_graph2 (else_branch) | If_22 (If)]   <- branch named
19 [ONNX Layer: Div_14]                                                <- inside the Loop body, no hint
```

Layers 17..26 all came out of the `Loop` body, yet their metadata is
indistinguishable from a main-graph layer's. When debugging a control-flow model,
the boundary layer types are the only reliable landmark.
