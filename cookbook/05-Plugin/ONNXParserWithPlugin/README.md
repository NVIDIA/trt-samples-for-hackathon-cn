# ONNX Parser and Plugin

+ Example of combinating the usage of model from ONNX and plugin.

+ Steps to run.

```bash
make build
python3 main.py
```

| Case | What it shows |
| --- | --- |
| `case_normal` | parse a ready-made ONNX file containing an `AddScalar` node |
| `case_pytorch_to_onnx_to_trt` | **produce** that ONNX file from a PyTorch model |
| `case_the_three_traps` | the three ways it goes wrong, each provoked in a fresh process |

## `case_pytorch_to_onnx_to_trt` — PyTorch → ONNX → TensorRT with a plugin node

`case_normal` starts from a file that already contains the plugin node. This case is the missing
half: getting that node out of PyTorch in the first place.

Two mechanisms, adapted from `tensorrt_edgellm/onnx/onnx_custom_schemas.py` and
`dynamo_translations.py` of [TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)
(Apache-2.0), where they are what lets a Hugging Face checkpoint export with TensorRT plugin nodes
in it. Those files are 2467 and 1858 lines of transformer-specific schemas; this is the same two
mechanisms on one scalar add.

**1. Register an ONNX `OpSchema`** so the exporter knows what an `AddScalar` node looks like. Its
identity has to match the plugin exactly — op type `AddScalar`, domain `""` (the plugin's namespace
in `AddScalarPlugin.h` is the empty string), one float attribute `scalar`. Upstream fixes
`since_version` rather than tracking the opset, and makes the call idempotent because several
packages may register the same schema into one process; both are kept here.

**2. One entry in `custom_translation_table`** mapping `torch.ops.cookbook.add_scalar.default` to a
function that emits the node.

```txt
    Exported node: op_type='AddScalar' domain='' attributes={'scalar': 1.0}
    TensorRT [2. 2. 2. 2.] vs PyTorch eager [2. 2. 2. 2.]
```

## `case_the_three_traps`

All three were hit while writing the case above, and **none of them names its own cause**. Each
runs in a fresh subprocess, because all three depend on process-global state that cannot be undone.

**1. The schema must be registered before the export.** Upstream's docstring says exactly this —
*"Call register_... before `torch.onnx.export(..., optimize=True)` so custom nodes resolve"* — and
here is what skipping it costs:

```txt
    RESULT ConversionError <- ConversionError <- GraphConstructionError <- AttributeError <- SchemaError
```

The root cause is five levels down the chain. What you see at the top is `ConversionError`.

**2. The translation must be a plain function, not `@onnxscript.script`.** A scripted function
becomes an ONNX *local function*, and inlining it then fails:

```txt
    RESULT PassError An error occurred when running the '<onnx_ir.passes.common.inliner.InlinePass ob...
```

The message names neither the function nor the operator. A plain function emits the node straight
into the graph and the problem disappears.

**3. A plugin path with no directory component silently does nothing.**

```txt
    RESULT AddScalarPlugin.so     -> NoneType   AddScalar registered: False
    RESULT ./AddScalarPlugin.so   -> PyCapsule  AddScalar registered: True
```

`IPluginRegistry.load_library` hands the string to `dlopen`, which searches the loader's library
paths and **not** the current directory. It returns `None` and raises nothing. The parser then
reports:

```txt
creator && "Plugin not found, are the plugin name, version, and namespace correct?"
```

— which points at three things that are all fine. This is why `plugin_file_list` in `main.py` is
built from `Path(__file__).parent`, and why that is not merely tidiness.

Measured on TensorRT 11.3.0.99, torch 2.14.0a0, onnx 1.21.0, B200, 2026-10-12.
