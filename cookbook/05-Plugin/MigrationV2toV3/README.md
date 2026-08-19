# MigrationV2toV3

+ Migrate a Python plugin from the deprecated `IPluginV2DynamicExt` to `IPluginV3`.

+ `main.py` implements the same operator (`Y = scale * X`) twice, once per interface, sharing one
  CUDA kernel. Reading the two classes side by side *is* the migration guide - the mapping table
  below just indexes them.

+ `IPluginV2DynamicExt` has been deprecated since TensorRT-8.5 and is scheduled for removal in
  TensorRT-12, so existing plugins should move to `IPluginV3`.

+ Related examples: `05-Plugin/BasicExample` (V3, C++ kernel), `05-Plugin/BasicExample-V2-deprecated`
  (V2, C++ kernel), `05-Plugin/PythonPlugin` (the same op across several Python kernel backends),
  `05-Plugin/APIs` (a walk through every plugin API).

+ Steps to run.

```bash
python3 main.py
```

## Method-by-method mapping

| Concern            | `IPluginV2DynamicExt` (before)                              | `IPluginV3` (after)                                                     |
| ------------------ | ----------------------------------------------------------- | ----------------------------------------------------------------------- |
| Base classes       | `IPluginV2DynamicExt`                                        | `IPluginV3` + `IPluginV3OneCore` + `IPluginV3OneBuild` + `IPluginV3OneRuntime` |
| Creator base       | `IPluginCreator`                                             | `IPluginCreatorV3One`                                                    |
| Name attribute     | `plugin_type`                                                | `plugin_name`                                                            |
| Capability dispatch | not needed                                                  | `get_capability_interface(type)`                                         |
| Output datatype    | `get_output_datatype(index, input_types)` — one at a time    | `get_output_data_types(input_types)` — returns a list                    |
| Output shape       | `get_output_dimensions(index, inputs, expr_builder)` — one at a time | `get_output_shapes(inputs, shape_inputs, expr_builder)` — returns a list |
| Format support     | `in_out[pos]` is a `PluginTensorDesc`                        | `in_out[pos]` is a `DynamicPluginTensorDesc`, so use `in_out[pos].desc`  |
| Resource lifecycle | `initialize()` / `terminate()` / `destroy()`                 | none — acquire in `configure_plugin()` / `on_shape_change()`             |
| Per-context setup  | `attach_to_context()` / `detach_from_context()`              | `attach_to_context(context)` **returns** the per-context clone           |
| Tactics            | (not available)                                              | `get_valid_tactics()` / `set_tactic()`                                   |
| Serialization      | plugin implements `get_serialization_size()` + `serialize()`; creator implements `deserialize_plugin()` | plugin implements `get_fields_to_serialize()`; TensorRT serializes and re-creates through the creator |
| Create signature   | `create_plugin(name, field_collection)`                      | `create_plugin(name, field_collection, phase)`                           |
| Add to network     | `network.add_plugin_v2([input_tensors], plugin)`             | `network.add_plugin_v3([input_tensors], [input_shape_tensors], plugin)`  |

## What actually changes

+ **`enqueue()` does not change.** Both plugins call the same `launch_scale_kernel()`, so the kernel
  and its launch are untouched by the migration. Everything else is interface plumbing.

+ **Serialization is the biggest win.** V2 makes the plugin responsible for its own bytes: it has to
  report `get_serialization_size()` correctly, produce the bytes in `serialize()`, and the creator
  has to parse them back in `deserialize_plugin()` — three places to keep in sync, and a wrong size
  corrupts the engine. V3 replaces all of it with `get_fields_to_serialize()`; TensorRT stores those
  fields and rebuilds the plugin through the creator's `create_plugin(..., phase=RUNTIME)`. There is
  no deserialization code in the V3 half of `main.py` at all.

  Each case in `main.py` runs twice — once building from scratch, once reloading the serialized
  engine — so both deserialization paths are actually exercised rather than just described.

+ **One class becomes four interfaces.** `IPluginV3` is the object itself; the three capability
  interfaces split what used to be one flat class into build-time (`IPluginV3OneBuild`), runtime
  (`IPluginV3OneRuntime`) and identity (`IPluginV3OneCore`) concerns. They may live in separate
  classes; `main.py` lets one object implement all three, so `get_capability_interface()` just
  returns `self`.

+ **Batched getters.** V2 asked for one output's datatype/shape per call; V3 asks once and gets a
  list. Plugins with several outputs get noticeably shorter.

+ **`attach_to_context` returns instead of mutates.** In V3 it hands back the per-context clone,
  which makes the per-context state explicit rather than hidden in `self`.
