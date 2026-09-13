# 10 - Export Engine to ONNX

Export a TensorRT engine plan to an ONNX file so it can be explored visually in
[Netron](https://netron.app). This is the cookbook re-implementation of
`trt-engine-explorer`'s `graphing.OnnxGraph` / `make_onnx_tensor`.

One ONNX node is emitted per engine layer, wired together by tensor names; the
engine's input/output bindings become the ONNX graph inputs/outputs.

> **Note:** the exported ONNX is a *visualization aid*, not a runnable model. It
> uses TensorRT layer names (e.g. `Reformat`) as op types, which Netron displays
> fine but `onnx.checker` / onnxruntime will reject.

## Running

```bash
python3 ../get_data.py   # build the engine + export JSON (needs a GPU; skipped if present)
python3 main.py          # export the engine to ONNX (no GPU required)
netron engine_model.onnx # (optional) explore the graph in Netron
```

## Output

+ `engine_model.onnx` - the engine graph as an ONNX file.
+ `case_export_onnx` prints the node count, graph inputs/outputs and the first few nodes.
