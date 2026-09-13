# Onnx Graphsurgeon

+ A python library for ONNX compute graph edition, which different from the library *onnx*.

+ Installation: `pip install nvidia-pyindex onnx-graphsungeon`

+ Document [Link](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html)

+ The example code here refers to the NVIDIA official repository about TensorRT tools [Link](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/examples).

+ Function:
  + Modify metadata/node / tensor / weight data of compute graph.
  + Modify subgraph: Add / Delete / Replace / Isolate
  + Optimize: constant folding / topological sorting / removing useless layers.

+ `11` to `13` are the toolbox distilled from a hand-written ChatGLM-6B pipeline
  ([`../../99-Todo/chatglm-6b.md`](../../99-Todo/chatglm-6b.md)) -- the three things graph surgery is
  actually used for on a real model, as opposed to the API tour in `01` to `10`:

  + **`11_mark_output_to_bisect.py`** -- turn "the FP16 answer is wrong" into a node name.
    `mark_graph_output` cuts the graph down to one node and makes its output the graph output, so
    walking forward finds the first node where the precisions diverge. The example's FP16 graph
    silently returns zeros -- no `inf`, no `nan` to warn you -- and the bisect pins the overflow on
    the middle `Mul`, not on the node that produced the wrong final value.
  + **`12_constant_table.py`** -- constant folding cannot touch a subgraph that depends on a runtime
    tensor, even when that tensor only ever takes values from a bounded set. A rotary position
    embedding is the classic case: evaluate `cos(position * inv_freq)` once on the host for every
    reachable position and it becomes a single `Gather`. 4 nodes per step become 1; the catch is
    that the table size is now a hard ceiling the original graph did not have.
  + **`13_merge_two_models.py`** -- join a body and a separately exported head, both by lifting a
    weight out of the second file as a `gs.Constant` and by transplanting its nodes, then append an
    `ArgMax` so the graph returns a class index instead of a score vector.

+ `14_fold_exporter_subgraphs.py` is the same idea applied to **framework-exporter noise**, taken
  from `samples/python/onnx_packnet` in TensorRT-OSS. That sample hand-writes three rewrites; this
  one measures which of them are still worth writing:

  | Stage | ONNX nodes | TRT layers | max abs diff |
  | ----- | ---------: | ---------: | -----------: |
  | straight out of `torch.onnx.export` | 33 | 71 | – |
  | `+ fold_constants()` | 12 | 23 | 0 |
  | `+ GroupNormalization rewrite` | **7** | **13** | 3.3e-07 |

  + The `Pad` and `Resize` input subgraphs upstream rewrites by hand are **pure constant
    arithmetic**, so `fold_constants()` deletes them for free — 14 `Constant` nodes,
    `ConstantOfShape`, `Concat`, `Slice`, `Transpose`, `Cast` and two `Reshape`, all gone with no
    pattern matching. Two of the three upstream rewrites are obsolete.
  + The `GroupNorm` pattern (`Reshape` → `InstanceNormalization` → `Reshape` → `Mul` → `Add`)
    **survives folding**, because it depends on a runtime `Shape`. That one still needs a real
    rewrite, and it is the half worth reading.
  + Upstream replaces it with a `GroupNormalizationPlugin` node. **Do not copy that part**: the
    plugin has been unsupported on Blackwell and later since TensorRT 10.7, and its own README
    points at `INormalizationLayer`. This example emits the *native* ONNX `GroupNormalization`
    operator (opset 21) instead, which the parser turns into one `LayerType.NORMALIZATION`.
  + Upstream navigates the subgraph with chains like `node.i(1).i(0).i(0).i(0).i(0).i(0)` and needs
    three versions of that chain for three ranges of `torch.__version__`. Matching the pattern and
    reading values out of the matched nodes does not care which exporter produced the graph.

+ Steps to run. The scripts are numbered and meant to be read in order; each one is standalone, so
  any of them can also be run on its own.

```bash
python3 tests/run_tests.py --case 07-Tool/OnnxGraphSurgeon   # run all of them, from the cookbook root
python3 06_fold.py                                           # or just one, from this directory
```

  There is no `main.sh`: the ordered list lives in [`unit_test.yaml`](./unit_test.yaml), so the
  runner and a human reading the directory see exactly the same sequence.
