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
"""One engine layer-info JSON, four outputs: Graphviz `.gv` and its rendered picture, a Dagre HTML
page, and yEd GraphML.

`trtexec --exportLayerInfo` (or `IEngineInspector`) gives a JSON description of the *final* engine --
after fusion, after Myelin, with the layer names the profiler will use. Reading it as text stops
working at about thirty layers.

| Format | Needs | Good for |
| ------ | ----- | -------- |
| Graphviz `.gv` + `.svg` | `dot` for the picture; the `.gv` needs nothing | static images, scripting |
| **Dagre `.html`** | **nothing but a browser** | looking at it *now*, pan, zoom, hover |
| **yEd `.graphml`** | yEd (free, desktop) | hand-editing, re-layout, big graphs |

Both the `.gv` and the picture are written. `dot` is a *system* package -- `pip install graphviz`
provides only the bindings -- so it is genuinely optional: without it the `.gv` is still written and
only the image is skipped. Graphviz 2.43.0 was installed in this container on 2026-09-08, so the
image is rendered here; the HTML output exists for the machines where it is not.

`../trex/` analyses the same JSON in a dozen other ways. Its graph-drawing example used to be a
third implementation of this file's job and has been removed; what it drew, this draws, plus
per-tensor tooltips, dashed back-edges and multi-profile filtering.

Re-expressed from the idea in the internal `tools/engine_visualizer/plotEngine.py`. That file
carries a proprietary SPDX header, an internal email address and `gitlab-master.nvidia.com`
links; none of it was copied, and nothing internal appears here.
"""

import json
import re
import shutil
import subprocess
from collections import OrderedDict
from html import escape
from pathlib import Path
from xml.sax.saxutils import quoteattr

from tensorrt_cookbook import case_mark, cookbook_path

output_path = Path(__file__).parent
onnx_file = cookbook_path("00-Data", "model", "model-trained.onnx")
json_file = output_path / "data-layer_info.json"
graphviz_file = output_path / "result-engine.gv"
html_file = output_path / "result-engine.html"
graphml_file = output_path / "result-engine.graphml"
simple_graphviz_file = output_path / "result-engine-simple.gv"
profile_file = output_path / "data-profile.json"
multi_profile_json_file = output_path / "data-layer_info-multi_profile.json"

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

result = OrderedDict()

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

def load_graph(path: Path, profile_path: Path = None, profile_index: int = 0) -> tuple:
    """Return `(node_list, edge_list, binding_list)` from a layer-info JSON.

    A layer-info JSON is a **list of layers**, each naming the tensors it reads and writes. The
    edges are implicit: layer A feeds layer B when a name in A's `Outputs` appears in B's `Inputs`.
    Building that index is the whole conversion; everything after it is formatting.

    Two details that the obvious version of this function gets wrong:

    + **A tensor can have more than one producer.** An engine built from a model with a `Loop` keeps
      its carried tensor under one name and writes it from several layers, and one layer may even
      read and write the same name. Indexing `producer[name] = layer` keeps only the last writer and
      then drops the rest without a word; on a Loop engine that loses the entry edges - 3 of the 8
      real dependencies - and the picture no longer shows a loop at all. Here `producer` maps to a
      **list**, and self-loops are kept and marked.
    + **A multi-profile engine repeats its layers.** Layers belonging to optimization profile N > 0
      carry a `[profile N]` suffix, so drawing all of them shows the network two or three times
      over. `profile_index` selects one; the count of what was filtered is printed rather than
      silently dropped.
    """
    data = json.loads(path.read_text())
    layer_list = data["Layers"]
    binding_name_list = data.get("Bindings") or [t.get("Name", "") for t in data.get("I/O Tensors", [])]

    # ---- Optimization-profile filtering
    def belongs_to(name: str) -> bool:
        match = re.search(r"\[profile +([0-9]+)\]", name)
        return (match is None) if not profile_index else (match is not None and int(match.group(1)) == profile_index)

    n_before = len(layer_list)
    if any(re.search(r"\[profile +[0-9]+\]", layer["Name"]) for layer in layer_list):
        layer_list = [layer for layer in layer_list if belongs_to(layer["Name"])]
        binding_name_list = [name for name in binding_name_list if belongs_to(name)]
        print(f"    Engine has several optimization profiles: kept {len(layer_list)} of {n_before} layers "
              f"for profile {profile_index}. Pass `profile_index=N` to see another.")

    # ---- Optional per-layer latency, from `trtexec --exportProfile`
    latency_map = {}
    if profile_path is not None and profile_path.exists():
        for row in json.loads(profile_path.read_text()):
            if isinstance(row, dict) and "name" in row:
                latency_map[row["name"]] = float(row.get("averageMs", 0) or 0)

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
    consumed = {tensor["name"] for node in node_list for tensor in node["inputs"]}
    produced = {tensor["name"] for node in node_list for tensor in node["outputs"]}
    binding_list = []
    for binding_name in binding_name_list:
        if binding_name in produced:
            binding_list.append({"name": binding_name, "id": "out_" + binding_name, "direction": "output"})
        elif binding_name in consumed:
            binding_list.append({"name": binding_name, "id": "in_" + binding_name, "direction": "input"})
    return node_list, edge_list, binding_list

def short(name: str, limit: int = 28) -> str:
    return name if len(name) <= limit else "..." + name[-(limit - 3):]

def clean_layer_name(name: str) -> str:
    """Strip the noise TensorRT adds to a layer name, keeping what identifies it."""
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
        line_list.append(short(clean_layer_name(node["name"]), max_name_len))
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

def node_tooltip(node: dict) -> str:
    """Every field the JSON carried for this layer, for the formats that can show a tooltip."""
    line_list = [f"name: {node['name']}", f"type: {node['type']}"]
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
        line_list.append(f'  {node["id"]} [label="{label}", fillcolor="{colour_of(node)}", tooltip={quoteattr(node_tooltip(node))}{extra}];')
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

def rasterise(graphviz_path: Path, output_format: str = "svg") -> Path:
    """Render the `.gv` with the Graphviz `dot` binary, if there is one.

    Both outputs are wanted: the `.gv` is the source you can edit and re-render, the image is what
    you actually look at. `dot` is a **system** package (`apt-get install graphviz`) and the
    `graphviz` PyPI package only shells out to it, so the binary is genuinely optional - when it is
    missing the `.gv` is still written and only the picture is skipped.
    """
    image_path = graphviz_path.with_suffix(f".{output_format}")
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
        "title": node_tooltip(node),
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
            f'      <data key="d2">{escape(node_tooltip(node))}</data>',
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

def run_trtexec(argument_list: list) -> None:
    process = subprocess.run(["trtexec", *argument_list], capture_output=True, text=True, cwd=str(output_path))
    assert process.returncode == 0, process.stdout[-1500:]

@case_mark
def case_export_layer_info() -> None:
    """Get a layer-info JSON out of an engine, the way a user actually would."""
    run_trtexec([
        f"--onnx={onnx_file}",
        f"--exportLayerInfo={json_file}",
        f"--exportProfile={profile_file}",  # Optional, and what puts a latency on every node
        "--profilingVerbosity=detailed",  # Without this the names are useless
        "--iterations=10",  # `--exportProfile` needs the engine to actually run
    ])
    assert json_file.exists()
    data = json.loads(json_file.read_text())
    print(f"    {json_file.name}: {len(data['Layers'])} layers, top-level keys {list(data)}")
    print("    `--profilingVerbosity=detailed` is required; the default emits placeholder names.")
    return

@case_mark
def case_build_graph() -> None:
    """Turn the layer list into nodes, edges and bindings."""
    node_list, edge_list, binding_list = load_graph(json_file, profile_path=profile_file)
    result["node_list"], result["edge_list"], result["binding_list"] = node_list, edge_list, binding_list
    type_counter = OrderedDict()
    for node in node_list:
        type_counter[node["type"]] = type_counter.get(node["type"], 0) + 1
    precision_counter = OrderedDict()
    for edge in edge_list:
        key = edge["tensor"]["precision"]
        precision_counter[key] = precision_counter.get(key, 0) + 1
    n_back = sum(1 for edge in edge_list if edge["back_edge"])
    n_latency = sum(1 for node in node_list if node["latency"])
    print(f"    {len(node_list)} nodes, {len(edge_list)} edges ({n_back} back-edge), {len(binding_list)} bindings")
    print(f"    layer types    : {dict(sorted(type_counter.items()))}")
    print(f"    edge precisions: {dict(sorted(precision_counter.items()))}")
    print(f"    layers with a measured latency: {n_latency}")
    print("    Edges are implicit in the JSON: A -> B when a name in A's Outputs is in B's Inputs.")
    print("    A tensor may have several producers - keeping only the last would drop the loop edges.")
    return

@case_mark
def case_write_every_format() -> None:
    """Write all four outputs, and report what each one needs to be viewed."""
    node_list, edge_list, binding_list = result["node_list"], result["edge_list"], result["binding_list"]
    slowest = max((node for node in node_list if node["latency"]), key=lambda n: n["latency"], default=None)
    option = {"highlight_layers": [slowest["name"]] if slowest else []}
    if slowest:
        print(f"    Highlighting the slowest layer: {slowest['name']} ({slowest['latency']:.4f} ms)")

    write_graphviz(node_list, edge_list, binding_list, graphviz_file, **option)
    write_dagre_html(node_list, edge_list, binding_list, html_file, **option)
    write_graphml(node_list, edge_list, binding_list, graphml_file, **option)
    image_path = rasterise(graphviz_file, "svg")
    result["image_path"] = image_path

    entry_list = [(graphviz_file, "dot, or any Graphviz viewer"), (html_file, "any browser"), (graphml_file, "yEd")]
    if image_path is not None:
        entry_list.append((image_path, "any image viewer or browser"))
    for path, viewer in entry_list:
        print(f"    {path.name:<24}{path.stat().st_size:>8} B   view with: {viewer}")
    return

@case_mark
def case_simplified() -> None:
    """The same graph with everything optional turned off, for a large engine."""
    node_list, edge_list, binding_list = result["node_list"], result["edge_list"], result["binding_list"]
    option = {"b_display_edge_details": False, "b_display_bindings": False, "b_display_latency": False}
    write_graphviz(node_list, edge_list, binding_list, simple_graphviz_file, **option)
    print(f"    {simple_graphviz_file.name}: {simple_graphviz_file.stat().st_size} B "
          f"(vs {graphviz_file.stat().st_size} B with labels, bindings and latency)")
    return

@case_mark
def case_multi_profile() -> None:
    """An engine with two optimization profiles draws its network twice unless one is chosen.

    TensorRT still tags the layers of profile N > 0 with a `[profile N]` suffix, so the layer list
    holds one copy of the network per profile. Drawing all of them produces a picture with two
    disconnected halves that look like duplicates, which is not what anyone means by "the engine
    graph". `load_graph` keeps profile 0 by default - the same default as `trex` - and prints what
    it filtered rather than dropping it quietly.
    """
    run_trtexec([
        f"--onnx={onnx_file}",
        f"--exportLayerInfo={multi_profile_json_file}",
        "--profilingVerbosity=detailed",
        "--skipInference",
        "--profile=0",
        "--minShapes=x:1x1x28x28",
        "--optShapes=x:2x1x28x28",
        "--maxShapes=x:4x1x28x28",
        "--profile=1",
        "--minShapes=x:8x1x28x28",
        "--optShapes=x:32x1x28x28",
        "--maxShapes=x:64x1x28x28",
    ])
    n_total = len(json.loads(multi_profile_json_file.read_text())["Layers"])
    n_layer_per_profile = {}
    for profile_index in [0, 1]:
        node_list, edge_list, binding_list = load_graph(multi_profile_json_file, profile_index=profile_index)
        n_layer_per_profile[profile_index] = len(node_list)
        print(f"    profile {profile_index}: {len(node_list)} nodes, {len(edge_list)} edges, {len(binding_list)} bindings")
    assert sum(n_layer_per_profile.values()) >= n_total, "profile filtering lost layers that belong to neither"
    assert n_layer_per_profile[0] != n_layer_per_profile[1] or n_total == 2 * n_layer_per_profile[0], \
        "the two profiles produced identical counts, so the filter may not be doing anything"
    print(f"    {n_total} layers in the file, and neither profile alone accounts for all of them -")
    print("    drawing without filtering would show the network more than once.")
    return

@case_mark
def case_validate_output() -> None:
    """Check the two structured formats parse, rather than trusting the string building."""
    from xml.etree import ElementTree

    tree = ElementTree.parse(graphml_file)
    namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
    n_node = len(tree.getroot().findall(".//g:node", namespace))
    n_edge = len(tree.getroot().findall(".//g:edge", namespace))
    n_expected_node = len(result["node_list"]) + len(result["binding_list"])
    print(f"    GraphML parses: {n_node} nodes, {n_edge} edges")
    assert n_node == n_expected_node and n_edge == len(result["edge_list"]), "GraphML lost elements"

    text = html_file.read_text()
    assert "dagre-d3" in text and '"n0"' in text, "HTML is missing its data or its loader"
    assert "stroke-dasharray" in text, "HTML is missing the back-edge style"
    print("    HTML contains the node/edge arrays, the dagre-d3 loader and the precision legend: True")

    gv_text = graphviz_file.read_text()
    assert gv_text.startswith("digraph engine {") and gv_text.rstrip().endswith("}"), "DOT is malformed"
    n_edge_line = sum(1 for line in gv_text.splitlines() if "->" in line)
    n_coloured = sum(1 for line in gv_text.splitlines() if "->" in line and "color=" in line)
    assert n_coloured == n_edge_line, f"{n_edge_line - n_coloured} edges lost their colour"
    print(f"    DOT is well formed, and all {n_coloured} of its edges carry a precision colour: True")
    return

def main() -> None:
    case_export_layer_info()
    case_build_graph()
    case_write_every_format()
    case_simplified()
    case_multi_profile()
    case_validate_output()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
