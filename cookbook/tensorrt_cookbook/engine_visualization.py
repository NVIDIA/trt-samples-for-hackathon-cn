# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Draw an engine layer-info JSON as Graphviz, Dagre HTML or yEd GraphML.

`trtexec --exportLayerInfo` (or `IEngineInspector`) writes a JSON description of the *final* engine
- after fusion, after Myelin, with the layer names the profiler will use. Past about thirty layers
it stops being readable as text. This module turns that JSON into a node/edge list and formats it;
`07-Tool/EngineVisualization/` is the example that drives it.

Three writers, one graph
------------------------

| Writer                | Needs                          | Good for                              |
| --------------------- | ------------------------------ | ------------------------------------- |
| `write_graphviz`      | `dot` for the picture only     | static images, scripting              |
| `write_dagre_html`    | nothing but a browser          | looking at it *now*: pan, zoom, hover |
| `write_graphml`       | yEd (free, desktop)            | hand-editing the layout, big graphs   |

All three share `load_engine_graph`, the colour tables and the label helpers, so the three pictures
of one engine agree with each other.

Relation to the rest of the package
-----------------------------------

+ `utils_engine.export_engine_as_onnx` reads the **same** JSON into an ONNX file. It is deliberately
  a different reader: it renames tensors into single static assignment so the result is a DAG, which
  is what ONNX requires. The reader here keeps the cycles and marks them, because a picture of a
  Loop engine should still look like a loop.
+ `utils_engine_explorer` draws a graph too, from its own `EnginePlan` object rather than from the
  raw JSON. Note its `clean_layer_name` *escapes* Graphviz metacharacters, while
  `strip_layer_name_noise` here *removes* the decorations TensorRT adds to a layer name - same kind
  of input, opposite jobs, which is why the name is not shared.

Nothing here imports `tensorrt`: a layer-info JSON is just a file, and reading one should not
require the engine, the GPU, or even TensorRT to be installed.
"""

import json
import re
import shutil
import subprocess
from collections import namedtuple
from html import escape
from pathlib import Path
from xml.sax.saxutils import quoteattr

# One optimization profile's drawable graph. A plain tuple so it still unpacks three ways and still
# splats into the writers (`write_graphviz(*graph, path)`), named so the call sites read.
EngineGraph = namedtuple("EngineGraph", ("node_list", "edge_list", "binding_list"))

# Only the names a user of this module needs. The label and colour helpers below stay reachable as
# `tensorrt_cookbook.engine_visualization.<name>` but are kept out of the package namespace, which
# `__init__.py` fills with `import *` - `short` and `sanitise` are far too generic to live there,
# and `clean_layer_name` would silently shadow the differently-behaved one in
# `utils_engine_explorer`.
__all__ = [
    "TYPE_COLOUR",
    "DEFAULT_COLOUR",
    "PRECISION_COLOUR",
    "DEFAULT_PRECISION_COLOUR",
    "PRECISION_ALIAS",
    "precision_of",
    "EngineGraph",
    "load_engine_graph",
    "split_layers_by_profile",
    "compare_engine_profiles",
    "draw_engine_graph",
    "write_graphviz",
    "write_dagre_html",
    "write_graphml",
    "rasterise_graphviz",
]

# ================================================================ Colour tables

# Colour per layer type, shared by every writer so the outputs agree.
#
# The vocabulary is the one TensorRT 11 actually emits, not the capitalised names of the TRT 8/9 era.
# Measured on this container, an MNIST engine's layer types are
# `{custom_layer, kgen, correlation, maxpool, gemm}` and a Loop engine adds `{memset, add, cjmp}` -
# none of which appear in the older palettes, so a graph drawn with those came out almost entirely
# grey. The legacy names are kept below so engines built by an older TensorRT still colour.
TYPE_COLOUR = {
    # TensorRT 11 / Myelin vocabulary
    "kgen": "#C8E6C9",  # generated pointwise kernel, the most common node by far
    "gemm": "#BBDEFB",
    "correlation": "#BBDEFB",  # convolution, as Myelin names it
    "maxpool": "#FFE0B2",
    "avgpool": "#FFE0B2",
    "memset": "#F0F4C3",
    "add": "#DCEDC8",
    "cjmp": "#FFAB91",  # conditional branch: the loop back-edge control
    "shape_call": "#E1BEE7",
    "custom_layer": "#D1C4E9",
    "reformat": "#FFCDD2",
    # Pre-11 names, kept so older engines still colour
    "Convolution": "#BBDEFB",
    "CaskConvolution": "#BBDEFB",
    "PointWise": "#C8E6C9",
    "Pooling": "#FFE0B2",
    "Reformat": "#FFCDD2",
    "NoOp": "#ECEFF1",
    "Myelin": "#D1C4E9",
    "Constant": "#F0F4C3",
    "Shuffle": "#E1BEE7",
    "MatrixMultiply": "#BBDEFB",
    "ElementWise": "#DCEDC8",
    "SoftMax": "#F8BBD0",
    "Quantize": "#B2EBF2",
}
DEFAULT_COLOUR = "#F5F5F5"

# Colour per tensor precision, used for edges. This is the encoding that answers "where does the
# graph change precision", which is usually the first question asked of a mixed-precision engine.
PRECISION_COLOUR = {
    "FP32": "#D32F2F",
    "FP16": "#F57C00",
    "BF16": "#F9A825",
    "FP8": "#0288D1",
    "INT8": "#76B900",
    "INT4": "#7B1FA2",
    "FP4": "#C0A000",
    "INT32": "#9E9E9E",
    "INT64": "#757575",
    "BOOL": "#455A64",
}
DEFAULT_PRECISION_COLOUR = "#B0BEC5"

# TensorRT spells the same precision several ways depending on the field and the version
PRECISION_ALIAS = {
    "Float": "FP32",
    "FP32": "FP32",
    "Double": "FP64",
    "Half": "FP16",
    "FP16": "FP16",
    "BFloat16": "BF16",
    "BF16": "BF16",
    "Int8": "INT8",
    "INT8": "INT8",
    "Int32": "INT32",
    "INT32": "INT32",
    "Int64": "INT64",
    "INT64": "INT64",
    "Bool": "BOOL",
    "BOOL": "BOOL",
    "FP8": "FP8",
    "Int4": "INT4",
    "INT4": "INT4",
    "FP4": "FP4",
}

def precision_of(data_type: str) -> str:
    """Normalise a `Datatype` string to the key used by `PRECISION_COLOUR`."""
    return PRECISION_ALIAS.get(data_type, data_type or "Unknown")

# ================================================================ Read the JSON into a graph

def parse_tensor(raw: dict) -> dict:
    """Everything the JSON records about one tensor, not just its name.

    `Datatype` and `Format` are both present in every layer-info entry; keeping them is what makes
    precision-coloured edges and shape-labelled edges possible at all.
    """
    data_type = raw.get("Datatype", "")
    return {
        "name": raw.get("Name", ""),
        "shape": raw.get("Dimensions", []),
        "dtype": data_type,
        "format": raw.get("Format", ""),
        "precision": precision_of(data_type),
    }

PROFILE_TAG_RE = re.compile(r"\[profile +([0-9]+)\]")
MYELIN_REGION_RE = re.compile(r"_myl([0-9]+)_[0-9]+$")

def split_layers_by_profile(layer_list: list) -> list:
    """Group an engine's layers by optimization profile: `[[layers of profile 0], [of 1], ...]`.

    A multi-profile engine holds **one complete copy of the network per profile**, and splitting
    them apart is harder than the obvious `[profile N]` test makes it look.

    Measured on a two-profile MNIST engine, 24 layers in all:

    ```txt
        Myelin region myl0, untagged        12 layers   <- profile 0, complete
        Myelin region myl1, [profile 1]      6 layers   <- profile 1, tagged half
        Myelin region myl1, untagged         6 layers   <- profile 1, but carries no tag
    ```

    **Only half of profile 1's layers carry the tag.** The other six are Myelin-generated
    (`__myl_Move`, `__myl_Resh`, `__myl_CastReshCast`, …) and are named exactly like their profile-0
    counterparts apart from the region number. Keying on the tag alone therefore gets it wrong in
    both directions: "profile 0" picks up 18 layers, six of which belong to profile 1, and
    "profile 1" returns a six-layer fragment with no bindings attached.

    What is reliable is the **Myelin region** in the `_myl<region>_<index>` suffix: a region belongs
    to one profile, so the tagged layers in a region identify the whole region. Layers with no such
    suffix - a non-Myelin engine - fall back to their own tag, and anything untagged is profile 0.
    A single-profile engine has one region, no tags, and comes back as a one-element list.
    """
    region_profile = {}
    for layer in layer_list:
        region = MYELIN_REGION_RE.search(layer.get("Name", ""))
        tag = PROFILE_TAG_RE.search(layer.get("Name", ""))
        if region is not None and tag is not None:
            region_profile[region.group(1)] = int(tag.group(1))

    def profile_of(name: str) -> int:
        region = MYELIN_REGION_RE.search(name)
        if region is not None:
            return region_profile.get(region.group(1), 0)
        tag = PROFILE_TAG_RE.search(name)
        return int(tag.group(1)) if tag is not None else 0

    n_profile = max((profile_of(layer.get("Name", "")) for layer in layer_list), default=0) + 1
    bucket_list = [[] for _ in range(n_profile)]
    for layer in layer_list:
        bucket_list[profile_of(layer.get("Name", ""))].append(layer)
    return bucket_list

def load_engine_graph(path: Path, profile_path: Path = None) -> "EngineGraph":
    """Return the `EngineGraph` of **optimization profile 0** from a layer-info JSON.

    A multi-profile engine holds one complete copy of the network per profile, and this draws the
    first. The others are not built and not returned, because **they produce the same picture**:

    + The profile's min/opt/max shape ranges are not recorded in this JSON at all - every dynamic
      dimension reads `-1` in every copy - so the copies have the same topology and the same shapes.
    + What does differ is `TacticName`, and only that. On a two-profile MNIST engine most of the 12
      paired layers picked a different kernel, sensibly so: `node_max_pool2d` took
      `..._PROPAGATE_NAN_3D` for the batch 1-4 profile and `..._PROPAGATE_NAN_2D` for the batch
      8-64 one. But `TacticName` reaches the writers through `node_tooltip` alone - it sets no
      colour, no label and no edge - so drawing profile 1 gives byte-identical geometry with
      different hover text. Measured: of the 35 lines by which the two `.gv` files differ, every
      one is inside a `tooltip=` attribute.

    The other profiles still have to be **identified**, though, which is the part that is easy to
    get wrong - see `split_layers_by_profile`. Without it all the copies are drawn at once, which is
    the "why is my engine graph two disconnected networks" symptom. Anything that wants the other
    profiles - comparing their tactics, for instance - uses `split_layers_by_profile` on the raw
    layer list rather than building graphs nobody draws.

    `EngineGraph` is a `(node_list, edge_list, binding_list)` triple, so it unpacks three ways and
    splats into the writers: `write_graphviz(*graph, path)`.

    A layer-info JSON is a **list of layers**, each naming the tensors it reads and writes. The
    edges are implicit: layer A feeds layer B when a name in A's `Outputs` appears in B's `Inputs`.
    Building that index is the whole conversion; everything after it is formatting.

    One detail the obvious version of this function gets wrong: **a tensor can have more than one
    producer.** An engine built from a model with a `Loop` keeps its carried tensor under one name
    and writes it from several layers, and one layer may even read and write the same name. Indexing
    `producer[name] = layer` keeps only the last writer and then drops the rest without a word; on a
    Loop engine that loses the entry edges - 3 of the 8 real dependencies - and the picture no longer
    shows a loop at all. Here `producer` maps to a **list**, and self-loops are kept and marked.
    """
    data = json.loads(path.read_text())
    binding_name_list = data.get("Bindings") or [t.get("Name", "") for t in data.get("I/O Tensors", [])]

    # ---- Optional per-layer latency, from `trtexec --exportProfile`
    latency_map = {}
    if profile_path is not None and profile_path.exists():
        for row in json.loads(profile_path.read_text()):
            if isinstance(row, dict) and "name" in row:
                latency_map[row["name"]] = float(row.get("averageMs", 0) or 0)

    layer_list = split_layers_by_profile(data["Layers"])[0]

    node_list = []
    producer = {}  # tensor name -> [index of every layer that writes it]
    for index, layer in enumerate(layer_list):
        name = layer.get("Name", f"layer_{index}")
        node_list.append({
            "id": f"n{index}",
            "name": name,
            "type": layer.get("LayerType", "Unknown"),
            "tactic": layer.get("TacticName", ""),
            "metadata": layer.get("Metadata", ""),
            "stream_id": layer.get("StreamId", ""),
            "constants": layer.get("Constants", ""),
            "latency": latency_map.get(name, None),
            "inputs": [parse_tensor(t) for t in layer.get("Inputs", [])],
            "outputs": [parse_tensor(t) for t in layer.get("Outputs", [])],
        })
        for tensor in layer.get("Outputs", []):
            producer.setdefault(tensor.get("Name", ""), []).append(index)

    edge_list = []
    for index, node in enumerate(node_list):
        for tensor in node["inputs"]:
            for source in producer.get(tensor["name"], []):
                edge_list.append({
                    "source": f"n{source}",
                    "target": f"n{index}",
                    "label": tensor["name"],
                    "tensor": tensor,
                    # A back-edge is a write that happens at or after the layer reading it: the loop
                    # closing. Marked rather than dropped, so a viewer can draw it differently.
                    "back_edge": source >= index,
                })

    # ---- Engine input / output bindings, drawn as terminal nodes
    #
    # `Bindings` names the engine I/O tensors the way profile 0 spells them, which is why this works
    # without any further filtering. A profile N > 0 carries the same tensor under a tagged name -
    # profile 1's input `x` is `x [profile 1]` - so a reader that drew another profile would have to
    # try both spellings before deciding a binding is absent.
    consumed = {tensor["name"] for node in node_list for tensor in node["inputs"]}
    produced = {tensor["name"] for node in node_list for tensor in node["outputs"]}
    binding_list = []
    for binding_name in binding_name_list:
        if binding_name in produced:
            binding_list.append({"name": binding_name, "id": "out_" + binding_name, "direction": "output"})
        elif binding_name in consumed:
            binding_list.append({"name": binding_name, "id": "in_" + binding_name, "direction": "input"})
    return EngineGraph(node_list, edge_list, binding_list)

# ================================================================ Labels and colours

def short(name: str, limit: int = 28) -> str:
    return name if len(name) <= limit else "..." + name[-(limit - 3):]

def strip_layer_name_noise(name: str) -> str:
    """Strip the noise TensorRT adds to a layer name, keeping what identifies it.

    Not called `clean_layer_name`: `utils_engine_explorer` exports a function of that name which
    *escapes* Graphviz metacharacters instead of removing them, and having both in the package
    namespace would mean whichever module `__init__.py` imports last wins.
    """
    for noise in ("|| ", "[Convolution]", "[Fully Connected]"):
        name = name.replace(noise, "")
    return name.strip()

def colour_of(node: dict) -> str:
    for key, value in TYPE_COLOUR.items():
        if node["type"] == key:
            return value
    return DEFAULT_COLOUR

def edge_colour_of(edge: dict) -> str:
    return PRECISION_COLOUR.get(edge["tensor"]["precision"], DEFAULT_PRECISION_COLOUR)

def node_label(node: dict, b_display_name: bool = True, b_display_latency: bool = True, max_name_len: int = 28) -> str:
    line_list = []
    if b_display_name:
        line_list.append(short(strip_layer_name_noise(node["name"]), max_name_len))
    line_list.append(f"[{node['type']}]")
    if b_display_latency and node["latency"]:
        line_list.append(f"{node['latency']:.4f} ms")
    return "\n".join(line_list)

def edge_label(edge: dict, b_display_details: bool = True) -> str:
    """Shape and format, not just the tensor name - the name alone rarely says anything."""
    if not b_display_details:
        return ""
    tensor = edge["tensor"]
    return f"{tensor['shape']}\n{tensor['precision']}"

def sanitise(text: str) -> str:
    """Drop the control characters XML 1.0 forbids.

    TensorRT separates the entries of a `Metadata` field with `\x1f` (unit separator), so a layer
    fused from two ONNX nodes reads `[ONNX Layer: node_conv2d]\x1f[ONNX Layer: node_relu]`. That
    byte is legal in JSON and illegal in XML, and `xml.sax.saxutils.escape` does not touch it - it
    only handles `<`, `>` and `&` - so writing it straight into GraphML produces a file that yEd and
    `ElementTree` both refuse with `not well-formed (invalid token)`.
    """
    return "".join(" | " if character == "\x1f" else character for character in text if character >= " " or character in "\n\t")

def node_tooltip(node: dict, b_display_details: bool = True) -> str:
    """Every field the JSON carried for this layer, for the formats that can show a tooltip.

    `b_display_details=False` keeps the name and the type and drops the rest. The tooltip is where
    nearly all of the bytes are - `Constants` alone can run to several hundred characters per
    layer - so this is what makes the stripped-down drawing actually small rather than merely
    plainer. The toggle is separate from `b_display_latency`, which only governs the node *label*.
    """
    line_list = [f"name: {node['name']}", f"type: {node['type']}"]
    if not b_display_details:
        return sanitise("\n".join(line_list))
    for key in ("tactic", "metadata", "stream_id", "constants"):
        if node[key]:
            line_list.append(f"{key}: {node[key]}")
    if node["latency"]:
        line_list.append(f"latency: {node['latency']:.4f} ms")
    for direction in ("inputs", "outputs"):
        for tensor in node[direction]:
            line_list.append(f"{direction[:-1]}: {tensor['name']} {tensor['shape']} {tensor['dtype']} {tensor['format']}")
    return sanitise("\n".join(line_list))

# ================================================================ Writers

def write_graphviz(node_list: list, edge_list: list, binding_list: list, path: Path, **kwargs) -> None:
    line_list = ["digraph engine {", "  rankdir=TB;", '  node [shape=rectangle, style=filled, fontname="Helvetica", fontsize=10];', '  edge [fontname="Helvetica", fontsize=8];']
    highlight_set = set(kwargs.get("highlight_layers", []) or [])
    for node in node_list:
        label = node_label(node, kwargs.get("b_display_name", True), kwargs.get("b_display_latency", True)).replace("\n", "\\n")
        extra = ', penwidth=4, color="red"' if node["name"] in highlight_set else ""
        line_list.append(f'  {node["id"]} [label="{label}", fillcolor="{colour_of(node)}", tooltip={quoteattr(node_tooltip(node, kwargs.get("b_display_tooltip_details", True)))}{extra}];')
    if kwargs.get("b_display_bindings", True):
        for binding in binding_list:
            line_list.append(f'  "{binding["id"]}" [label="{short(binding["name"])}", shape=oval, fillcolor="#CFD8DC"];')
    for edge in edge_list:
        style = ', style=dashed, constraint=false' if edge["back_edge"] else ""
        label = edge_label(edge, kwargs.get("b_display_edge_details", True)).replace("\n", "\\n")
        line_list.append(f'  {edge["source"]} -> {edge["target"]} [label="{label}", color="{edge_colour_of(edge)}"{style}];')
    if kwargs.get("b_display_bindings", True):
        for binding in binding_list:
            for node in node_list:
                if binding["direction"] == "input" and binding["name"] in [t["name"] for t in node["inputs"]]:
                    line_list.append(f'  "{binding["id"]}" -> {node["id"]} [color="#90A4AE"];')
                if binding["direction"] == "output" and binding["name"] in [t["name"] for t in node["outputs"]]:
                    line_list.append(f'  {node["id"]} -> "{binding["id"]}" [color="#90A4AE"];')
    line_list.append("}")
    path.write_text("\n".join(line_list) + "\n")

def rasterise_graphviz(graphviz_path: Path, output_format: str = "svg", image_path: Path = None) -> Path:
    """Render the `.gv` with the Graphviz `dot` binary, if there is one.

    Both outputs are wanted: the `.gv` is the source you can edit and re-render, the image is what
    you actually look at. `dot` is a **system** package (`apt-get install graphviz`) and the
    `graphviz` PyPI package only shells out to it, so the binary is genuinely optional - when it is
    missing the `.gv` is still written and only the picture is skipped, and `None` is returned.
    """
    image_path = Path(image_path) if image_path is not None else graphviz_path.with_suffix(f".{output_format}")
    if shutil.which("dot") is None:
        print(f"    [SKIP] `dot` is not on PATH, so {image_path.name} was not rendered ({graphviz_path.name} is still written)")
        return None
    subprocess.run(["dot", f"-T{output_format}", str(graphviz_path), "-o", str(image_path)], check=True)
    return image_path

def write_dagre_html(node_list: list, edge_list: list, binding_list: list, path: Path, **kwargs) -> None:
    """A self-contained HTML page: the only format that needs nothing but a browser."""
    highlight_set = set(kwargs.get("highlight_layers", []) or [])
    node_json = json.dumps([{
        "id": node["id"],
        "label": node_label(node, kwargs.get("b_display_name", True), kwargs.get("b_display_latency", True), 34),
        "colour": colour_of(node),
        "highlight": node["name"] in highlight_set,
        "title": node_tooltip(node, kwargs.get("b_display_tooltip_details", True)),
    } for node in node_list] + [{
        "id": binding["id"],
        "label": short(binding["name"], 34),
        "colour": "#CFD8DC",
        "highlight": False,
        "title": f"engine {binding['direction']} binding",
    } for binding in binding_list if kwargs.get("b_display_bindings", True)])
    binding_edge_list = []
    if kwargs.get("b_display_bindings", True):
        for binding in binding_list:
            for node in node_list:
                if binding["direction"] == "input" and binding["name"] in [t["name"] for t in node["inputs"]]:
                    binding_edge_list.append({"source": binding["id"], "target": node["id"], "label": "", "colour": "#90A4AE", "back": False})
                if binding["direction"] == "output" and binding["name"] in [t["name"] for t in node["outputs"]]:
                    binding_edge_list.append({"source": node["id"], "target": binding["id"], "label": "", "colour": "#90A4AE", "back": False})
    edge_json = json.dumps([{
        "source": edge["source"],
        "target": edge["target"],
        "label": edge_label(edge, kwargs.get("b_display_edge_details", True)).replace("\n", " "),
        "colour": edge_colour_of(edge),
        "back": edge["back_edge"],
    } for edge in edge_list] + binding_edge_list)
    legend = "".join(f'<span style="background:{colour};color:#fff;padding:1px 6px;margin-right:4px;border-radius:3px">{name}</span>' for name, colour in PRECISION_COLOUR.items())

    path.write_text(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TensorRT engine graph</title>
<script src="https://d3js.org/d3.v5.min.js"></script>
<script src="https://unpkg.com/dagre-d3@0.6.4/dist/dagre-d3.min.js"></script>
<style>
 body {{ margin: 0; font-family: Helvetica, Arial, sans-serif; background: #FAFAFA; }}
 #header {{ padding: 8px 12px; background: #263238; color: #fff; font-size: 14px; }}
 #legend {{ padding: 6px 12px; font-size: 11px; background: #37474F; color: #ECEFF1; }}
 .node rect {{ stroke: #546E7A; stroke-width: 1px; }}
 .node.highlight rect {{ stroke: red; stroke-width: 4px; }}
 .edgePath path {{ stroke-width: 1.4px; fill: none; }}
 .edgePath.back path {{ stroke-dasharray: 5 3; }}
 text {{ font-size: 11px; }}
</style></head>
<body>
<div id="header">TensorRT engine graph &mdash; drag to pan, scroll to zoom, hover a node for its full record</div>
<div id="legend">edge colour = tensor precision: {legend}<span style="margin-left:10px">dashed = back-edge (loop)</span></div>
<svg id="svg" width="100%" height="900"><g/></svg>
<script>
const nodeData = {node_json};
const edgeData = {edge_json};
const g = new dagreD3.graphlib.Graph().setGraph({{rankdir: "TB", nodesep: 30, ranksep: 40}});
nodeData.forEach(n => g.setNode(n.id, {{
    label: n.label, style: "fill: " + n.colour, labelStyle: "font-size: 11px",
    class: n.highlight ? "highlight" : "", title: n.title
}}));
edgeData.forEach(e => g.setEdge(e.source, e.target, {{
    label: e.label, curve: d3.curveBasis,
    style: "stroke: " + e.colour, arrowheadStyle: "fill: " + e.colour,
    class: e.back ? "back" : ""
}}));
const render = new dagreD3.render();
const svg = d3.select("#svg"), inner = svg.select("g");
const zoom = d3.zoom().on("zoom", () => inner.attr("transform", d3.event.transform));
svg.call(zoom);
render(inner, g);
inner.selectAll("g.node").append("title").text(d => g.node(d).title);
</script></body></html>
""")

def write_graphml(node_list: list, edge_list: list, binding_list: list, path: Path, **kwargs) -> None:
    """yEd GraphML: the one format whose layout a human can rearrange by hand."""
    highlight_set = set(kwargs.get("highlight_layers", []) or [])
    line_list = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:y="http://www.yworks.com/xml/graphml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">',
        '  <key for="node" id="d0" yfiles.type="nodegraphics"/>',
        '  <key for="edge" id="d1" yfiles.type="edgegraphics"/>',
        '  <key for="node" id="d2" attr.name="description" attr.type="string"/>',
        '  <key for="edge" id="d3" attr.name="description" attr.type="string"/>',
        '  <graph edgedefault="directed" id="G">',
    ]
    for node in node_list:
        label = escape(node_label(node, kwargs.get("b_display_name", True), kwargs.get("b_display_latency", True), 34).replace("\\n", "\n"))
        border = '#FF0000" type="line" width="4.0' if node["name"] in highlight_set else '#546E7A" type="line" width="1.0'
        line_list += [
            f'    <node id={quoteattr(node["id"])}>',
            '      <data key="d0"><y:ShapeNode>',
            '        <y:Geometry height="42.0" width="220.0"/>',
            f'        <y:Fill color={quoteattr(colour_of(node))} transparent="false"/>',
            f'        <y:BorderStyle color="{border}"/>',
            f'        <y:NodeLabel alignment="center" autoSizePolicy="content" fontSize="11">{label}</y:NodeLabel>',
            '      </y:ShapeNode></data>',
            f'      <data key="d2">{escape(node_tooltip(node, kwargs.get("b_display_tooltip_details", True)))}</data>',
            '    </node>',
        ]
    if kwargs.get("b_display_bindings", True):
        for binding in binding_list:
            line_list += [
                f'    <node id={quoteattr(binding["id"])}>',
                '      <data key="d0"><y:ShapeNode>',
                '        <y:Geometry height="32.0" width="160.0"/>',
                '        <y:Fill color="#CFD8DC" transparent="false"/>',
                '        <y:Shape type="ellipse"/>',
                f'        <y:NodeLabel fontSize="10">{escape(short(binding["name"], 34))}</y:NodeLabel>',
                '      </y:ShapeNode></data>',
                f'      <data key="d2">engine {binding["direction"]} binding</data>',
                '    </node>',
            ]
    for index, edge in enumerate(edge_list):
        line_style = "dashed" if edge["back_edge"] else "line"
        line_list += [
            f'    <edge id="e{index}" source={quoteattr(edge["source"])} target={quoteattr(edge["target"])}>',
            '      <data key="d1"><y:PolyLineEdge>',
            f'        <y:LineStyle color={quoteattr(edge_colour_of(edge))} type="{line_style}" width="1.4"/>',
            '        <y:Arrows source="none" target="standard"/>',
            f'        <y:EdgeLabel fontSize="9">{escape(edge_label(edge, kwargs.get("b_display_edge_details", True)).replace(chr(10), " "))}</y:EdgeLabel>',
            '      </y:PolyLineEdge></data>',
            f'      <data key="d3">{escape(edge["label"])}</data>',
            '    </edge>',
        ]
    line_list += ["  </graph>", "</graphml>"]
    path.write_text("\n".join(line_list) + "\n")

# ================================================================ One call, every format

def draw_engine_graph(
    json_file: Path,
    profile_file: Path = None,
    *,
    graphviz_file: Path = None,
    image_file: Path = None,
    html_file: Path = None,
    graphml_file: Path = None,
    b_simplify: bool = False,
    b_validate: bool = True,
    b_print: bool = True,
) -> "EngineGraph":
    """Layer-info JSON in, pictures out: read, draw, and check what was drawn.

    `json_file` is what `trtexec --exportLayerInfo` (or `IEngineInspector`) wrote. `profile_file` is
    its optional `--exportProfile` companion, which is what puts a measured latency on every node
    and lets the slowest layer be outlined.

    **An output is produced only if its path is given**, so one call writes whichever of the four
    formats the caller actually wants:

    | Argument         | Output                                       |
    | ---------------- | -------------------------------------------- |
    | `graphviz_file`  | Graphviz `.gv`                               |
    | `image_file`     | the rendered picture; needs `dot` and a `.gv` |
    | `html_file`      | self-contained Dagre page                     |
    | `graphml_file`   | yEd GraphML                                   |

    `b_simplify` draws the stripped-down version - no edge labels, no bindings, no latency and no
    highlight - which is what stays readable on a large engine. It is a property of the drawing, not
    an extra output: call twice with different paths to get both.

    `b_validate` parses the structured formats back and checks them, because string-building XML and
    HTML is easy to get subtly wrong in ways that still render. See `07-Tool/EngineVisualization/`
    for the two faults this has actually caught.

    Returns the `EngineGraph` that was drawn, so the caller can go on inspecting it.
    """
    graph = load_engine_graph(json_file, profile_path=profile_file)
    node_list, edge_list, binding_list = graph

    if b_print:
        n_back = sum(1 for edge in edge_list if edge["back_edge"])
        n_latency = sum(1 for node in node_list if node["latency"])
        type_counter, precision_counter = {}, {}
        for node in node_list:
            type_counter[node["type"]] = type_counter.get(node["type"], 0) + 1
        for edge in edge_list:
            key = edge["tensor"]["precision"]
            precision_counter[key] = precision_counter.get(key, 0) + 1
        print(f"    {len(node_list)} nodes, {len(edge_list)} edges ({n_back} back-edge), {len(binding_list)} bindings")
        print(f"    layer types    : {dict(sorted(type_counter.items()))}")
        print(f"    edge precisions: {dict(sorted(precision_counter.items()))}")
        print(f"    layers with a measured latency: {n_latency}")

    # ---- Drawing options
    if b_simplify:
        option = {"b_display_edge_details": False, "b_display_bindings": False, "b_display_latency": False, "b_display_tooltip_details": False}
    else:
        # The slowest layer is the one the reader is looking for, so it is outlined rather than left
        # to be found by comparing the numbers on thirteen nodes.
        slowest = max((node for node in node_list if node["latency"]), key=lambda node: node["latency"], default=None)
        option = {"highlight_layers": [slowest["name"]] if slowest else []}
        if b_print and slowest:
            print(f"    Highlighting the slowest layer: {slowest['name']} ({slowest['latency']:.4f} ms)")

    # ---- Write whichever formats were asked for
    written = []
    if graphviz_file is not None:
        write_graphviz(node_list, edge_list, binding_list, Path(graphviz_file), **option)
        written.append((Path(graphviz_file), "dot or any Graphviz viewer"))
        if image_file is not None:
            image_path = rasterise_graphviz(Path(graphviz_file), Path(image_file).suffix.lstrip(".") or "svg", Path(image_file))
            if image_path is not None:
                written.append((image_path, "any image viewer or browser"))
    elif image_file is not None:
        raise ValueError("`image_file` renders the `.gv`, so `graphviz_file` has to be given as well")
    if html_file is not None:
        write_dagre_html(node_list, edge_list, binding_list, Path(html_file), **option)
        written.append((Path(html_file), "any browser"))
    if graphml_file is not None:
        write_graphml(node_list, edge_list, binding_list, Path(graphml_file), **option)
        written.append((Path(graphml_file), "yEd"))

    if b_print:
        for path, viewer in written:
            print(f"    {path.name:<34}{path.stat().st_size:>8} B   view with: {viewer}")

    if b_validate:
        _validate_output(graph, option, graphviz_file, html_file, graphml_file, b_print)
    return graph

def _validate_output(graph: "EngineGraph", option: dict, graphviz_file: Path, html_file: Path, graphml_file: Path, b_print: bool) -> None:
    """Parse the structured formats back, for the faults that render anyway.

    Only the formats that were actually written are checked, so this follows whichever outputs the
    caller asked for - and it has to follow the drawing `option` too: `b_display_bindings=False`
    leaves the terminal nodes out, so counting on them being there turns a correct simplified
    drawing into a failed assertion.
    """
    node_list, edge_list, binding_list = graph
    n_expected_node = len(node_list) + (len(binding_list) if option.get("b_display_bindings", True) else 0)

    if graphml_file is not None:
        from xml.etree import ElementTree

        tree = ElementTree.parse(graphml_file)
        namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
        n_node = len(tree.getroot().findall(".//g:node", namespace))
        n_edge = len(tree.getroot().findall(".//g:edge", namespace))
        assert n_node == n_expected_node and n_edge == len(edge_list), "GraphML lost elements"
        if b_print:
            print(f"    GraphML parses: {n_node} nodes, {n_edge} edges")

    if html_file is not None:
        text = Path(html_file).read_text()
        assert "dagre-d3" in text and '"n0"' in text, "HTML is missing its data or its loader"
        assert "stroke-dasharray" in text, "HTML is missing the back-edge style"
        if b_print:
            print("    HTML contains the node/edge arrays, the dagre-d3 loader and the precision legend: True")

    if graphviz_file is not None:
        gv_text = Path(graphviz_file).read_text()
        assert gv_text.startswith("digraph engine {") and gv_text.rstrip().endswith("}"), "DOT is malformed"
        n_edge_line = sum(1 for line in gv_text.splitlines() if "->" in line)
        n_coloured = sum(1 for line in gv_text.splitlines() if "->" in line and "color=" in line)
        assert n_coloured == n_edge_line, f"{n_edge_line - n_coloured} edges lost their colour"
        if b_print:
            print(f"    DOT is well formed, and all {n_coloured} of its edges carry a precision colour: True")

def compare_engine_profiles(json_file: Path, b_print: bool = True) -> dict:
    """Split a layer-info JSON by optimization profile and report how the copies differ.

    The answer, for every multi-profile engine seen so far, is "only in the tactic", which is why
    `load_engine_graph` builds profile 0 and leaves the rest. Drawing the others would give the same
    geometry with different hover text, so this reports the difference instead of drawing it:

    + The profile's min/opt/max shape ranges are **not in this JSON at all** - every dynamic
      dimension reads `-1` in every copy - so the copies share a topology and a set of shapes.
    + `TacticName` does differ, and sensibly: on a two-profile MNIST engine `node_max_pool2d` takes
      `..._PROPAGATE_NAN_3D` for the batch 1-4 profile and `..._PROPAGATE_NAN_2D` for the batch
      8-64 one. Which layers differ is a property of one build and moves between runs.

    Layers are paired across profiles by their **logical name** - the layer name with the
    `[profile N]` tag and the `_myl<region>_<index>` suffix removed - since those two are exactly
    what the copies are spelled differently by.

    Returns `{"n_layer", "layers_per_profile", "paired_name_list", "tactic_per_profile"}`, where
    `tactic_per_profile` maps a logical name to one tactic per profile and holds only the names
    whose tactics are not all equal.
    """
    layer_list = json.loads(Path(json_file).read_text())["Layers"]
    layers_per_profile = split_layers_by_profile(layer_list)
    assert sum(len(item) for item in layers_per_profile) == len(layer_list), "the split lost or duplicated layers"

    def logical_name(name: str) -> str:
        return re.sub(r" *\[profile +[0-9]+\]", "", MYELIN_REGION_RE.sub("", name))

    tactic_of = [{logical_name(layer["Name"]): layer.get("TacticName", "") for layer in item} for item in layers_per_profile]
    paired_name_list = sorted(set.intersection(*(set(mapping) for mapping in tactic_of))) if tactic_of else []
    differ = {name: [mapping[name] for mapping in tactic_of] for name in paired_name_list if len({mapping[name] for mapping in tactic_of}) > 1}

    if b_print:
        print(f"    {len(layer_list)} layers in the file, {len(layers_per_profile)} optimization profiles")
        for profile_index, profile_layer_list in enumerate(layers_per_profile):
            print(f"      profile {profile_index}: {len(profile_layer_list):3d} layers")
        if len(layers_per_profile) > 1:
            print(f"    the {len(layers_per_profile)} copies are the same topology with the same shapes: every dynamic")
            print("    dimension reads -1 in all of them, and the profile's min/opt/max is not in this file.")
            print(f"    what differs is the tactic: {len(differ)} of the {len(paired_name_list)} paired layers picked a different kernel,")
            for name, tactic_list in differ.items():
                print(f"      {name:22s} {' | '.join(_around_first_difference(tactic_list))}")
            print("    and a tactic reaches the drawing through the node tooltip alone, so a second")
            print("    picture would be the first one with different hover text.")
    return {
        "n_layer": len(layer_list),
        "layers_per_profile": layers_per_profile,
        "paired_name_list": paired_name_list,
        "tactic_per_profile": differ,
    }

def _around_first_difference(text_list: list, width: int = 12) -> list:
    """A window of each string around the first character at which they disagree.

    Two tactic names can agree for sixty-odd characters and differ only at the end -
    `..._PROPAGATE_NAN_3D` against `..._PROPAGATE_NAN_2D` - so printing the start of each would look
    like evidence against the very claim being made. `zip` stops at the shortest, so when one name
    is a prefix of another the first difference is the end of the common prefix.
    """
    head = next((index for index, column in enumerate(zip(*text_list)) if len(set(column)) > 1), min(len(text) for text in text_list))
    return ["…" + text[max(0, head - width):head + width] for text in text_list]
