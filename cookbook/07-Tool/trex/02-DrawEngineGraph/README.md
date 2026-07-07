# 02 - Draw Engine Graph

Draw a TensorRT engine plan as a directed graph, in the spirit of
`trt-engine-explorer`'s `graphing.DotGraph` / `to_dot` and the
`utils/draw_engine.py` script. This cookbook version is a compact re-implementation
built directly on `EnginePlan`:

+ one **node per layer**, colored by layer type (`layer_colormap`), labeled with
  the layer name, type and latency;
+ one **edge per data dependency** (producer layer -> consumer layer), colored by
  the tensor precision (`precision_colormap`) and labeled with its shape/dtype;
+ the engine's input/output **bindings** are drawn as gray terminal nodes.

## Requirements

The Graphviz **`dot` binary** must be installed (the python `graphviz` package
just shells out to it):

```bash
sudo apt-get install -y graphviz
```

## Running

```bash
python3 ../get_data.py   # build the engine + export JSON (needs a GPU; skipped if present)
python3 main.py          # render the engine graph (needs the graphviz `dot` binary)
```

## Figures produced

| File                            | Content                                                       |
| ------------------------------- | ------------------------------------------------------------- |
| `engine_graph.png`              | full graph: layers by type, edges by precision, binding nodes |
| `engine_graph.svg`              | the same graph as a zoomable vector image                     |
| `engine_graph_simple.png`       | simplified graph: no edge labels / bindings / latency         |
| `engine_graph_highlight.png`    | full graph with the slowest layer highlighted in red          |

The rendering options are exposed as keyword arguments of `build_engine_graph` /
`render_engine_graph` (see `tensorrt_cookbook/utils_engine_explorer.py`):
`display_layer_names`, `display_latency`, `display_edge_details`,
`display_bindings`, `display_constants`, `highlight_layers`, `max_name_len`.
