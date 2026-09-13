# Engine Visualization

+ One engine layer-info JSON, five outputs: Graphviz DOT and the picture it renders, a
  self-contained Dagre HTML page, yEd GraphML, and ONNX for Netron.

+ Steps to run.

```bash
python3 main.py            # .gv + .svg, .html, .graphml
python3 export_as_onnx.py  # .onnx
```

The ONNX writer is a separate script rather than another case in `main.py` because it is a
different kind of conversion. The other writers build a node and edge list and format it; that one
builds a real `onnx_graphsurgeon` graph, so it keeps everything the JSON says about every layer and
every tensor instead of just the topology.

`trtexec --exportLayerInfo` describes the **final** engine — after fusion, after Myelin, with the
names the profiler will use. Reading it as text stops working at about thirty layers.

| Format | Needs | Good for |
| ------ | ----- | -------- |
| Graphviz `.gv` + `.svg` | `dot` for the picture; the `.gv` needs nothing | publication-quality static images, scripting |
| **Dagre `.html`** | **nothing but a browser** | looking at it now; pan, zoom, hover for the full layer record |
| **yEd `.graphml`** | yEd (free desktop app) | hand-editing, re-layout, large graphs |
| **ONNX** | Netron | the most information kept, and other ONNX tooling can read it |

**Both the `.gv` and the picture are written.** The `.gv` is the source you can edit and re-render;
the `.svg` is what you actually look at. `dot` is a *system* package (`apt-get install graphviz`)
and `pip install graphviz` does **not** provide it — that package only shells out to the binary — so
when `dot` is missing the `.gv` is still written and only the image is skipped:

```txt
    [SKIP] `dot` is not on PATH, so result-engine.svg was not rendered (result-engine.gv is still written)
```

## What is drawn

Every writer shares one palette and one set of toggles, so the four outputs agree.

+ **Nodes are filled by layer type.** The vocabulary is the one TensorRT 11 actually emits —
  `kgen`, `gemm`, `correlation`, `maxpool`, `memset`, `add`, `cjmp`, `custom_layer` — not the
  capitalised `Convolution` / `Shuffle` / `PointWise` names of the TRT 8/9 era. That distinction is
  not cosmetic: measured on this container, an MNIST engine's types are
  `{custom_layer, kgen, correlation, maxpool, gemm}`, of which the older palettes coloured at most
  two, so a graph drawn with them came out almost entirely grey. The legacy names are kept as well,
  so an engine from an older TensorRT still colours.
+ **Edges are coloured by tensor precision** (FP32 red, FP16 orange, INT8 green, FP8 blue, …). This
  is the encoding that answers "where does this engine change precision", usually the first question
  asked of a mixed-precision build. The HTML page carries a legend.
+ **Edge labels carry shape and precision**, not the tensor name. The name of a fused Myelin tensor
  says nothing; `[1, 32, 14, 14] FP32` says what is flowing.
+ **Per-layer latency**, when `trtexec --exportProfile` output is passed alongside the layer info.
  The slowest layer is outlined in red automatically.
+ **Engine input/output bindings** are drawn as grey oval terminal nodes.
+ **Back-edges are dashed**, so a loop reads as a loop. See the control-flow section below.
+ **Every field the JSON carried** — `TacticName`, `Metadata`, `StreamId`, `Constants`, and the
  dtype/format/shape of each tensor — is in the node tooltip: `tooltip=` in the `.gv`, a
  `description` key in the GraphML, a `<title>` in the HTML.

The toggles (`b_display_name`, `b_display_latency`, `b_display_edge_details`, `b_display_bindings`,
`highlight_layers`, `max_name_len`) are keyword arguments of every writer; `case_simplified` shows
the stripped-down version that stays readable on a large engine.

## Multi-profile engines are filtered, not merged

An engine built with several optimization profiles holds **one copy of the network per profile**;
TensorRT tags the layers of profile N > 0 with a `[profile N]` suffix. Drawing all of them gives a
picture with two disconnected halves that look like duplicates. `load_graph` keeps profile 0 by
default — the same default as `trex` — and says what it did:

```txt
    Engine has several optimization profiles: kept 18 of 24 layers for profile 0. Pass `profile_index=N` to see another.
    profile 0: 18 nodes, 15 edges, 3 bindings
    profile 1:  6 nodes,  4 edges, 0 bindings
```

Neither profile alone accounts for all 24 layers, which is the point: without filtering you are not
looking at one engine graph.

## `NoOp` layers are kept on purpose

`trex` removes `NoOp` layers and rewires their producers and consumers around them. This does not,
and the omission is deliberate: **a no-op is itself a finding.** A `NoOp` in a built engine usually
marks a reformat or a copy that the builder could not eliminate, and hiding it makes the graph
prettier while removing the evidence. It also conflicts with the rule the ONNX writer follows — do
not silently change the structure of what you were asked to draw.

## The conversion is one index

A layer-info JSON is a **list of layers**, each naming the tensors it reads and writes. There is no
edge list — edges are implicit: layer A feeds layer B when a name in A's `Outputs` appears in B's
`Inputs`. Building that producer index is the entire conversion; the writers are formatting.

The index maps a tensor name to **a list** of producers, not to one. That is not a detail: a single
producer per name is what makes the naive version drop the edges of a loop. See the control-flow
section.

On the MNIST engine: 13 nodes, 11 edges, layer types
`{correlation: 2, custom_layer: 2, gemm: 1, kgen: 6, maxpool: 2}` — note these are *engine* layer
types after fusion, not the ONNX ops you started with.

## Two flags that are easy to get wrong

+ **`--profilingVerbosity=detailed` is required at build time.** Without it the JSON contains
  placeholder names and the graph is unreadable.
+ **`--buildOnly` no longer exists — it is `--skipInference`.** As elsewhere in trtexec, an unknown
  option is answered by printing the *entire help text* and exiting non-zero, with no mention of
  which argument was wrong. Note the latency case cannot use it at all: `--exportProfile` needs the
  engine to actually run.

## Output is validated, not assumed

String-building an XML or HTML file is easy to get subtly wrong, so `case_validate_output` parses the
GraphML back with `ElementTree`, checks the node and edge counts survive, confirms the HTML carries
its data arrays, its loader and the back-edge style, and asserts that **every** edge in the DOT
carries a colour. It has earned its place twice:

+ **GraphML rejected as `not well-formed (invalid token)`.** TensorRT separates the entries of a
  `Metadata` field with `\x1f` (unit separator), so a layer fused from two ONNX nodes reads
  `[ONNX Layer: node_conv2d]\x1f[ONNX Layer: node_relu]`. That byte is legal in JSON and illegal in
  XML 1.0, and `xml.sax.saxutils.escape` does not touch it — it only handles `<`, `>` and `&`.
  `sanitise()` now strips control characters.
+ **Edges silently losing their colour in the DOT.** An edge label containing a real newline splits
  the edge definition across two lines. Graphviz tolerated it and still rendered, so nothing looked
  wrong; the check that counts coloured edges is what noticed. DOT labels now use the two-character
  `\n` escape.

Both were "runs fine, output is wrong" faults, which is the only kind this case can catch.

Re-expressed from the idea in the internal `tools/engine_visualizer/plotEngine.py`. That file carries
a proprietary SPDX header, an internal email address and `gitlab-master.nvidia.com` links; none of it
was copied and nothing internal appears here.

## Which format to reach for

Two writers here turn the same layer-info JSON into a picture. They are not redundant, but they are
not equally good at the same things either.

| | `export_as_onnx.py` | `main.py` |
| :-- | :-- | :-- |
| Output | `.onnx` | `.gv` + `.svg`, `.html`, `.graphml` |
| Viewer | Netron | image viewer, browser, yEd, `dot` |
| Per-layer attributes kept | **all of them** — `Constants`, `Metadata`, `StreamId`, `TacticName`, plus a synthesised `TensorInfo` and the optional `Latency` | all of them, in the node tooltip |
| Tensor dtype / shape | yes, on every tensor | yes, on every tensor |
| Visual encoding | none — Netron colours by operator category and you cannot change it | **nodes by layer type, edges by precision, slowest layer outlined, back-edges dashed** |
| Editing | **programmatic** (`onnx_graphsurgeon`) | **interactive** (yEd, on the GraphML) |
| Downstream tooling | **`onnx_outliner`, and anything else that reads ONNX** | none |
| Multi-profile engines | not filtered | **filtered, and says so** |
| Control flow (`Loop`) | all dependencies; cycle drawn as a cycle, or a DAG on request | all dependencies, back-edges dashed |

There used to be a third, `trex/02-DrawEngineGraph`, which rendered the same JSON with Graphviz.
Everything it drew is here now — nodes by layer type, edges by precision, per-layer latency, the
slowest layer outlined, the display toggles — so it was **removed** rather than kept as a fourth way
to make the same picture. Its palette could not simply be copied: it used the capitalised TRT 8/9
layer names, and on a TensorRT 11 engine almost every node fell through to grey.

Note the underlying `render_engine_graph` in `tensorrt_cookbook/utils_engine_explorer.py` stays,
because [`../trex/11-ProcessEnginePipeline/`](../trex/README.md) draws a graph as one step of its
pipeline. Only the example that existed solely to draw graphs is gone.

Two entries are easy to misread.

**The lack of visual encoding in the ONNX file is a limitation of the viewer, not of the format.**
Everything a renderer would need to colour by — dtype, format, memory location, measured latency —
is already in the file. Netron simply offers no control over its palette and draws no edge
attributes. A different renderer could colour that file by precision or by time without the exporter
changing at all.

**"Editable" means two different things.** GraphML in yEd is where you move nodes around by hand and
re-run a layout. ONNX is where you write a script that rewrites the graph — `onnx_graphsurgeon` is a
real graph-surgery API, which no other format here has. Neither replaces the other.

Per-layer latency is no longer a reason to prefer `trex/02`: both writers here take
`trtexec --exportProfile` output alongside the layer info. `main.py` puts the number on the node and
outlines the slowest layer in red, the same as `trex`; the ONNX file carries it as an attribute you
click to read.

## Control flow: how each writer handles it, and what it costs

An engine built from a model containing a `Loop` is the case where the three writers diverge, and
the difference is not the one it looks like from the outside.

**The engine has no subgraph.** TensorRT compiles a `Loop` into a flat instruction stream with a
back-edge — there is nothing nested to walk:

```txt
 4 memset   mov__mye64_myl0_4          out Recurrence 0 Output.2      <- initialise
 5 kgen     __myl_Move_myl0_5     in x out body_value_in.2            <- enter the loop
 6 kgen     __myl_Add_myl0_6      in body_value_in.2  out y           <- body
 7 add      ...ElementWise 2_to_add    in  Recurrence 0 Output.2
                                       out Recurrence 0 Output.2      <- self-loop: the counter
 8 kgen     __myl_Move_myl0_8     in y out body_value_in.2            <- back-edge
 9 cjmp     cbr___mye91_myl0_9         in ...                         <- conditional branch
```

Two consequences. A tensor name has **several producers** (`Recurrence 0 Output.2` is written by
layers 4 and 7; `body_value_in.2` by 5 and 8), and one layer **reads and writes the same name**.

**`main.py` and `trex/02` do not fail on this — they silently drop edges.** Both build a
`tensor name -> producer` index with the same two lines:

```python
producer[tensor_name] = index                  # a later write overwrites the earlier one
...
if source is not None and source != index:     # self-loops are discarded outright
```

On the engine above that leaves **5 of the 8 real dependencies**. The ones lost are exactly the
loop's entry edges — the initialisation from layer 4 and the first entry from layer 5 — so the
picture shows neither that this is a loop nor where its initial value comes from. No warning is
printed. (This is a general defect rather than a control-flow one: any engine reusing a tensor name
hits it. Fixing those two writers is tracked separately in `99-Todo`.)

**`export_as_onnx.py` keeps the cycle, because these files are read and never executed.** ONNX is
defined as single static assignment, so `onnx.checker` rejects a cyclic graph — but Netron draws it
correctly, as a pair of arrows between the two nodes. `case_cycle_is_viewable` builds a three-node
ONNX with a real cycle to keep that claim tested rather than remembered. Since a viewer that shows
the loop is worth more here than a checker that accepts the file, the default output leaves the
back-edge in place and the tensor names untouched.

**`b_break_cycle=True` gives the DAG instead**, for the cases that need one — anything built on
`onnx_graphsurgeon`, `onnx_outliner` included, raises `Cycle detected in graph!` on a cyclic file.
Two steps make the loop expressible without losing anything:

1. **SSA renaming.** Every write gets a name of its own (`...@0`, `...@2`) and every read is pointed
   at the write that reaches it. The cycle becomes a chain of versions, which is a DAG. The exporter
   already did this for Myelin scratch tensors; the predicate now also covers any tensor written
   more than once.
2. **An explicit `BackEdge` marker.** Renaming alone would leave the loop looking like a straight
   line, so one node per chain records where the cycle closes:

```txt
BackEdge node 'BackEdge-$$myelin$$(Unnamed Loop* 0)^Recurrence 0 Output .2_myl0'  domain=trt.engine
    inputs : ['...Recurrence 0 Output .2_myl0@2']
    OriginalName  = $$myelin$$(Unnamed Loop* 0)^Recurrence 0 Output .2_myl0
    Target        = ...Recurrence 0 Output .2_myl0@0
    Versions      = ['...@0', '...@2']
```

That node has no counterpart in the engine. It is a signpost saying "the cycle closes here",
carrying the whole version chain so the original structure can be read back.

**The DAG mode is harder to read, which is why it is not the default.** The `BackEdge` nodes are
dead ends: to see where one lands you have to open it, read the `Target` attribute and then find
that name elsewhere in the graph. Worse, removing the back-edge is often what splits the picture
into disconnected components, because that edge was the only thing joining them. The `@0` / `@2`
suffixes also lengthen names that TensorRT already made unwieldy.

| | default | `b_break_cycle=True` |
| :-- | :-- | :-- |
| back-edge | drawn as a cycle | `BackEdge` marker node |
| tensor names | unchanged | SSA versions, `...@0`, `...@2` |
| `onnx.checker` | rejects (cyclic) | passes |
| `onnx_graphsurgeon` / `onnx_outliner` | `Cycle detected in graph!` | works |
| reading the loop in Netron | direct | follow `Target` by name |

Engines **without** control flow are identical in both modes: no repeated tensor name means no
cycle, no renaming and no marker, so they pass `onnx.checker` either way.

| | dependencies represented | on failure |
| :-- | :-- | :-- |
| `export_as_onnx.py` | **8 of 8** (6 forward + 2 back-edges) | — |
| `main.py` | 5 of 8 | silent |
| `trex/02` | 5 of 8 | silent |

Verified on `model-loop.onnx` (a `Loop` and nothing else) and `model-for.onnx` (a `Loop` with an
`If` in its body, 4 back-edges); both pass `onnx.checker`. Engines without control flow produce zero
`BackEdge` nodes and are byte-for-byte what they were before.

**What this is not.** The output is a DAG that records where the cycle was, not an ONNX `Loop` with
a subgraph body. Reconstructing that would mean decompiling the flattened instruction stream —
finding the body boundaries from the `cjmp`, mapping recurrences to carried dependencies — and a
subtly wrong loop reconstruction is worse than an honest DAG, because it looks right. That remains
open in `99-Todo`.

## Related

+ [`../trex/`](../trex/README.md) — the full engine-analysis toolkit this complements.
+ [`../EnginePrinter/`](../EnginePrinter/README.md) — the engine's header and I/O tables, rather than
  its graph.
+ [`../trtexec/`](../trtexec/README.md) — where the JSON comes from.
+ [`../../04-Feature/Profiler/`](../../04-Feature/Profiler/README.md) — the same layer names, used for timing instead of topology.
