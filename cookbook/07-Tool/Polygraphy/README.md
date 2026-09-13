# Polygraphy - Client tool

+ CLI tool of polygraphy (deep learning model debugger).

+ Method of installation

```bash
pip install polygraphy
```

+ Features:
  + Do inference computation using multiple backends, including TensorRT, onnxruntime, TensorFlow etc.
  + Compare results of computation layer by layer among different backends.
  + Generate TensorRT engine from model file and serialize it as .trt file.
  + Print the detailed information of model, or view it as an interactive DAG (`inspect model --visual`).
  + Modify ONNX model, such as extracting subgraph, simplifying computation graph, stripping / reconstructing weights.
  + Analyze the failure of parsing ONNX model into TensorRT, and save the subgraphs that can / cannot be converted to TensorRT.
  + Bisect a failing model or a failing build down to the responsible node / tactic (`debug`).

+ One directory per subtool, each of them is driven by `main.sh`:

| Directory | Subtool | Note |
| --- | --- | --- |
| `Run/` | `polygraphy run` | inference + cross-backend comparison |
| `Convert/` | `polygraphy convert` | ONNX -> TensorRT engine / ONNX -> ONNX / TensorRT network -> ONNX-like |
| `Inspect/` | `polygraphy inspect` | `model` / `data` / `capability` / `sparsity`, plus the `--visual` viewer |
| `Check/` | `polygraphy check` | `lint` |
| `Surgeon/` | `polygraphy surgeon` | `sanitize` / `extract` / `insert` / `prune` / `weight-strip` / `weight-reconstruct` |
| `Template/` | `polygraphy template` | generate editable TensorRT network / config / ONNX-GS scripts |
| `Debug/` | `polygraphy debug` | `reduce` / `build` / `repeat` |
| `Data/` | `polygraphy data` | `merge` / `concat` |
| `Plugin/` | `polygraphy plugin` | match a subgraph against a plugin pattern and replace it |
| `MultiDevice/` | `polygraphy multi-device` + `template shard-hints` | rewrite a single-device ONNX into a CP / TP sharded one (the rewrite runs on one GPU, executing it does not) |
| `API/`, `More/` | Python API | `More/` holds 18 focused examples, see [`More/`](./More/) |
| `TensorRTRTX/` | (all subtools) | `POLYGRAPHY_USE_TENSORRT_RTX=1` points polygraphy at TensorRT-RTX; four places where the two backends are not interchangeable |
| `DebugWorkflow/` | (none -- scripts) | three workflows the CLI makes possible but does not provide: an error scatter plot, an auditable `debug reduce`, and a `--save-inputs` -> `trtexec --loadInputs` bridge |

+ **TensorRT 11 removed several precision features, and the Polygraphy CLI still advertises the
  corresponding options.** They only fail when a build is actually attempted:
  `--fp16` / `--bf16` / `--fp8` / `--int8` / `--precision-constraints` / `--layer-precisions` /
  `--save-tactics` / `--load-tactics`, and the whole `debug precision` subtool.
  `--tf32`, `--sparse-weights` and `--strongly-typed` still work.
  See [`More/06-Int8IsNowExplicit/`](./More/06-Int8IsNowExplicit/README.md),
  [`More/12-TacticsAndReproducibility/`](./More/12-TacticsAndReproducibility/README.md) and
  [`More/13-PerLayerPrecision/`](./More/13-PerLayerPrecision/README.md) for what to use instead.
