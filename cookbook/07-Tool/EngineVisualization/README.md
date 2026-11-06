# Engine Visualization

+ One engine layer-info JSON, five outputs: Graphviz DOT and the picture it renders, a
  self-contained Dagre HTML page, yEd GraphML, and ONNX for Netron.

+ Steps to run.

```bash
python3 main.py   # .gv + .svg, .html, .graphml, and .onnx
```

| Format                  | Needs                                          | Good for                                                      |
| ----------------------- | ---------------------------------------------- | ------------------------------------------------------------- |
| Graphviz `.gv` + `.svg` | `dot` for the picture; the `.gv` needs nothing | publication-quality static images, scripting                  |
| **Dagre `.html`**       | **nothing but a browser**                      | looking at it now; pan, zoom, hover for the full layer record |
| **yEd `.graphml`**      | yEd (free desktop app)                         | hand-editing, re-layout, large graphs                         |
| **ONNX**                | Netron                                         | the most information kept, and other ONNX tooling can read it |

**Both the `.gv` and the picture are written.** The `.gv` is the source you can edit and re-render;
the `.svg` is what you actually look at. `dot` is a *system* package (`apt-get install graphviz`)
and `pip install graphviz` does **not** provide it — that package only shells out to the binary — so
when `dot` is missing the `.gv` is still written and only the image is skipped:

```txt
    [SKIP] `dot` is not on PATH, so result-engine.svg was not rendered (result-engine.gv is still written)
```

Graphviz 2.43.0 was installed in this container on 2026-09-08, so the picture is rendered here. The
HTML output exists for the machines where it is not.

The ONNX output is a different kind of conversion from the other three: they build a node and edge
list and format it as text, while that one builds a real `onnx_graphsurgeon` graph, which is what
lets other ONNX tooling read the result. Both keep every field the JSON carried — see
[Which format to reach for](#which-format-to-reach-for) for where they actually differ. It used to
be a separate script, `export_as_onnx.py`, only because the example was too long; now that the
reading, drawing and checking live in the library it is a group of cases in `main.py`.

## One call for all of it

`draw_engine_graph` in `tensorrt_cookbook/engine_visualization.py` reads the JSON, draws, and checks
what it drew. **An output is produced only if its path is given**, so asking for fewer files is a
matter of leaving arguments out:

```python
draw_engine_graph(
    json_file,                      # trtexec --exportLayerInfo
    profile_file,                   # optional: --exportProfile, puts a latency on every node
    graphviz_file=...,              # .gv
    image_file=...,                 # the rendered picture; needs `dot` and a graphviz_file
    html_file=...,                  # self-contained Dagre page
    graphml_file=...,               # yEd
)
draw_engine_graph(json_file, graphviz_file=simple_file, b_simplify=True)   # the stripped-down one
```

`b_simplify` is a property of the *drawing* — no edge labels, no bindings, no latency, no highlight
and a name-and-type-only tooltip — not an extra output, so both versions are two calls.
`b_validate=False` skips the checks and `b_print=False` the narration.

The tooltip is where nearly all of the bytes are, so trimming it is what makes the simplified
drawing small rather than merely plainer. Measured on the MNIST engine, average tooltip 769
characters down to 43:

| | full | simplified |
| --- | ---: | ---: |
| `.gv` | 12276 B | **2220 B** |
| `.html` | 15847 B | 5826 B |
| `.graphml` | 21748 B | 10147 B |

Note the separate toggles underneath: `b_display_latency` governs the node *label*, and
`b_display_tooltip_details` the hover text. Setting only the first leaves every `TacticName`,
`Constants` blob and tensor record in the file.

Underneath are the pieces, all exported, for anyone who wants to hold the graph:
`load_engine_graph`, `split_layers_by_profile`, `write_graphviz`, `write_dagre_html`,
`write_graphml`, `rasterise_graphviz`. `main.py` keeps only what the example is about: getting the
JSON out of an engine, and what each output format is worth. The ONNX writer is
`export_engine_as_onnx` in `tensorrt_cookbook/utils_engine.py`.

## Where the JSON comes from

`trtexec --exportLayerInfo` describes the **final** engine — after fusion, after Myelin, with the
names the profiler will use. Reading it as text stops working at about thirty layers.

Two entry points, one string. `trtexec` has no layer-info implementation of its own: it creates an
`IEngineInspector` over the engine and writes `getEngineInformation(kJSON)` to the file
(`samples/common/sampleReporting.cpp`). Measured on TensorRT 11.0.0.114, for one and the same plan
file the output of

```bash
trtexec --loadEngine=model.plan --exportLayerInfo=layer_info.json
```

and of

```python
engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
```

are **byte-for-byte identical**, with and without an execution context bound. See
[`../../04-Feature/EngineInspector/`](../../04-Feature/EngineInspector/README.md) for the Python
side; either one can feed the writers here.

The scripts here use `trtexec` anyway, and the reason is the one place the two entry points do not
overlap: the latency cases also need `--exportProfile`, which measures layers at run time and is not
something the inspector reports. Everything else is a matter of taste — trtexec needs no Python and
runs straight on a saved plan, while the inspector can additionally report one layer at a time.

What *does* change the content is the build-time profiling verbosity, which belongs to neither
entry point; see the flags section below.

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
`highlight_layers`, `max_name_len`) are keyword arguments of every writer; the second `.gv` that
`main.py` writes, `result-engine-simple.gv`, is the stripped-down version that stays readable on a
large engine.

## The cases, and why they are split that way

`main.py` has one case per **engine or per question**, never one per step — the steps of a workflow
pass locals to each other, so splitting them would mean passing a module-level dictionary around.

**Pictures**

+ `case_draw_every_format` — get a layer-info JSON out of an engine, then one `draw_engine_graph`
  call for all four outputs and a second for the stripped-down `.gv`. The reading, drawing and
  checking are in the library; what is left here is the trtexec invocation and the arguments.
+ `case_multi_profile` — a **different engine**, hence a separate case. The number of optimization
  profiles is fixed at build time, so there is nothing to split unless trtexec is asked for more
  than one profile up front. It builds its own engine with two profiles, writes its own JSON, and
  draws `result-engine-multi_profile.gv` / `.svg` from profile 0 — passing no `html_file` or
  `graphml_file`, so those two formats are simply not written.

**ONNX**

+ `case_export_as_onnx` — the same MNIST engine as an ONNX file, exported twice: with the profile
  and without. The second export is there to *assert* that the attribute set does not depend on it,
  rather than claiming so in a comment.
+ `case_export_onnx_corpus` — three models that have each broken the exporter at some point.
+ `case_export_onnx_control_flow` — an engine with a `Loop`, exported both ways, plus the hand-built
  cyclic reference that justifies the default. `build_cyclic_reference_onnx` writes it and checks
  that `onnx.checker` still rejects a cycle, so the premise is re-tested rather than remembered.

## Multi-profile engines: profile 0 is drawn, the rest are identified and dropped

An engine built with several optimization profiles holds **one complete copy of the network per
profile**, so drawing the layer list as it comes puts several networks in one picture.
`load_engine_graph` returns the `EngineGraph` of **profile 0** and does not build the others:

```python
graph = load_engine_graph(json_file)   # always profile 0
write_graphviz(*graph, path)           # an (node_list, edge_list, binding_list) triple
```

The other copies still have to be **identified** — that is the only way to leave them out — which
is what `split_layers_by_profile` does, on the raw layer list and without building anything:

```txt
    24 layers in the file, 2 optimization profiles
      profile 0:  12 layers
      profile 1:  12 layers
```

### Telling the profiles apart is not `[profile N]`

The obvious test — keep the layers tagged `[profile N]` — gets it wrong in **both** directions.
Measured on the engine above:

```txt
    Myelin region myl0, untagged        12 layers   <- profile 0, complete
    Myelin region myl1, [profile 1]      6 layers   <- profile 1, tagged half
    Myelin region myl1, untagged         6 layers   <- profile 1, but carries no tag
```

Only **half** of profile 1's layers carry the tag; the other six are Myelin-generated
(`__myl_Move`, `__myl_Resh`, `__myl_CastReshCast`, …) and are named exactly like their profile-0
counterparts apart from the region number. Keying on the tag gives "profile 0 = 18 layers", six of
which actually belong to profile 1, and "profile 1 = 6 layers" with **no bindings attached** —
the tagged fragment never touches `x` or `y` because profile 1 reads them as `x [profile 1]`.

What is reliable is the **Myelin region** in the `_myl<region>_<index>` suffix: a region belongs to
one profile, so the tagged layers in a region identify the whole region. That gives the 12 / 12
split above, and each profile keeps its three bindings. `split_layers_by_profile` does this.

### The copies differ in a way no picture shows

This is why only profile 0 is built.

The per-profile min/opt/max shape ranges are **not in this JSON at all** — every dynamic dimension
reads `-1` in every copy — so the profiles cannot be told apart by topology or by shape. What does
differ is the **tactic**, and sensibly so: `node_max_pool2d` takes `..._PROPAGATE_NAN_3D` for the
batch 1–4 profile and `..._PROPAGATE_NAN_2D` for the batch 8–64 one, and `node_linear` takes
`sm80_xmma_gemm_...` against `sm86_xmma_gemm_...`.

`TacticName` reaches the writers through **`node_tooltip` alone** — it sets no colour, no label and
no edge. Node fill comes from `LayerType`, edge colour from tensor precision, the label from the
name, type and latency. Drawing profile 1 therefore gives the same geometry with different hover
text: measured, of the lines by which the two `.gv` files differ, **every one is inside a
`tooltip=` attribute**.

`compare_engine_profiles` does this comparison - splitting by profile, pairing layers by their
logical name (the name with the `[profile N]` tag and the `_myl<region>_<index>` suffix removed) and
printing a window around the first differing character of each tactic pair. `case_multi_profile`
calls it and draws once. Which
layers differ is a property of one build and moves between runs, so nothing asserts a count.

The picture is not one connected chain either way: profile 0 falls into 2 connected components and
the single-profile engine into 3, because the `custom_layer` nodes that set up the L2 cache policy
hang off on their own. That has nothing to do with profiles.

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

String-building an XML or HTML file is easy to get subtly wrong, so `draw_engine_graph` parses the
GraphML back with `ElementTree`, checks the node and edge counts survive, confirms the HTML carries
its data arrays, its loader and the back-edge style, and asserts that **every** edge in the DOT
carries a colour — each for whichever formats were actually written. It has earned its place twice:

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

## Which format to reach for

Two writers here turn the same layer-info JSON into a picture. They are not redundant, but they are
not equally good at the same things either.

|                           | the ONNX cases                                                                                                                  | the picture cases                                                                      |
| :------------------------ | :------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------- |
| Output                    | `.onnx`                                                                                                                         | `.gv` + `.svg`, `.html`, `.graphml`                                                    |
| Viewer                    | Netron                                                                                                                          | image viewer, browser, yEd, `dot`                                                      |
| Per-layer attributes kept | **all of them** — `Constants`, `Metadata`, `StreamId`, `TacticName`, plus a synthesised `TensorInfo` and the optional `Latency` | all of them, in the node tooltip                                                       |
| Tensor dtype / shape      | yes, on every tensor                                                                                                            | yes, on every tensor                                                                   |
| Visual encoding           | none — Netron colours by operator category and you cannot change it                                                             | **nodes by layer type, edges by precision, slowest layer outlined, back-edges dashed** |
| Editing                   | **programmatic** (`onnx_graphsurgeon`)                                                                                          | **interactive** (yEd, on the GraphML)                                                  |
| Downstream tooling        | **`onnx_outliner`, and anything else that reads ONNX**                                                                          | none                                                                                   |
| Multi-profile engines     | not filtered                                                                                                                    | **filtered, and says so**                                                              |
| Control flow (`Loop`)     | all dependencies; cycle drawn as a cycle, or a DAG on request                                                                   | all dependencies, back-edges dashed                                                    |

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

**The naive reader drops edges here, silently.** Indexing `producer[tensor_name] = index` keeps
only the last writer, and `if source != index` throws away the self-loop on top of that:

```python
producer[tensor_name] = index                  # a later write overwrites the earlier one
...
if source is not None and source != index:     # self-loops are discarded outright
```

What is lost is exactly the loop's entry edges — the initialisation and the first entry — so the
picture shows neither that this is a loop nor where its initial value comes from, and no warning is
printed. This is a general defect rather than a control-flow one: any engine reusing a tensor name
hits it. `trex/02` had it; `load_engine_graph` does not, because it maps a name to a **list** of
producers and marks the self-loops instead of discarding them.

**The ONNX export keeps the cycle, because these files are read and never executed.** ONNX is
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

|                                       | default                    | `b_break_cycle=True`           |
| :------------------------------------ | :------------------------- | :----------------------------- |
| back-edge                             | drawn as a cycle           | `BackEdge` marker node         |
| tensor names                          | unchanged                  | SSA versions, `...@0`, `...@2` |
| `onnx.checker`                        | rejects (cyclic)           | passes                         |
| `onnx_graphsurgeon` / `onnx_outliner` | `Cycle detected in graph!` | works                          |
| reading the loop in Netron            | direct                     | follow `Target` by name        |

Engines **without** control flow are identical in both modes: no repeated tensor name means no
cycle, no renaming and no marker, so they pass `onnx.checker` either way.

|                     | dependencies represented                    | on failure |
| :------------------ | :------------------------------------------ | :--------- |
| ONNX export         | **all of them**, cycles kept or SSA-versioned | —          |
| picture writers     | **all of them**, back-edges dashed           | —          |
| `trex/02`           | the back-edges dropped                       | silent     |

Measured on this build of `model-loop.onnx` (a `Loop` and nothing else): the picture writers give
11 nodes and 9 edges, 2 of them back-edges; the ONNX export gives 11 nodes with the same 2 cycles,
and 2 `BackEdge` markers under `dag_onnx_file`. `model-for.onnx` (a `Loop` with an `If` in its body)
gives 4. The counts move with the build, which is why the cases assert that the cycle count and the
marker count agree rather than pinning either. Engines without control flow produce zero `BackEdge`
nodes and both outputs pass `onnx.checker`.

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

---

Re-expressed from the idea in the internal file. That file carries a proprietary SPDX header, an
internal email address and `gitlab-master.nvidia.com` links; none of it was copied and nothing
internal appears here.
