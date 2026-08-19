# Candidates from the ChatGLM-6B TensorRT pipeline

Source: `/work/chatglm-6b` (856 lines of Python + a `Makefile`), a customer-facing
PyTorch → ONNX → TensorRT workflow for ChatGLM-6B, written against **TensorRT 8.6** in
`nvcr.io/nvidia/pytorch:23.01-py3`. Read 2026-08-28. Nothing was run — the box has no 6B checkpoint
and the code targets a TensorRT version three majors behind the cookbook.

This predates TensorRT-LLM's dominance and does by hand what TRT-LLM later productised. That is
exactly why it is worth mining: **every trick is visible in ~850 lines instead of buried in a
framework**, and most of them are not model-specific at all.

## The pipeline, in four stages

```txt
configuration.py  ──> configuration.yaml        # one generated config, consumed by every stage
exportONNX.py     ──> onnx/TranAllInOne/model.onnx  (28-layer transformer, 115 I/O tensors)
                      onnx/lm/model.onnx            (lm_head alone)
surgeon.py        ──> onnx/TranAllInOne/model-V1.onnx   # 4 rewrite passes, then constant folding
buildEngine.py    ──> plan/TranAllInOne.plan            # ~12 GiB FP16 engine
main.py           ──> prefill step + autoregressive decode loop
```

`Makefile` wires them with real file-level dependencies, so a change to `configuration.py`
invalidates exactly what it should. ~25 min end to end at max-batch-size 1.

---

## What is worth taking

| # | Technique | Why it generalises | Cookbook target | Prio |
| - | --------- | ------------------ | --------------- | ---- |
| C1 | **Fuse the whole KV cache into one I/O tensor** | 28 layers × (K,V) = **56 inputs + 56 outputs → 1 + 1** | 03-Workflow (new LLM-shaped leaf) / 04-Feature | **High** |
| C2 | **Bind the KV input and the KV output to the same address** | the cache grows in place; no copy, no ping-pong | 08-Advance or 04-Feature | **High** |
| C3 | **Move post-processing inside the engine** — slice last position → MatMul(lm_head) → Softmax → ArgMax, so the engine returns a **token id** | kills a 130528-wide D2H copy *per generated token* | 03-Workflow | **High** |
| C4 | **Export by patching the model's own `forward`** instead of writing an export script | captures the graph you actually run, with real KV inputs | 03-Workflow / 07-Tool | **High** |
| C5 | **`markGraphOutput` — a graph-surgery debugging primitive** | bisect an accuracy bug by truncating the ONNX at any node | 07-Tool/OnnxGraphSurgeon | **High** |
| C6 | **Precompute rotary cos/sin into constant tables at surgery time** | the general lesson: anything constant over the whole run should not be in the graph | 07-Tool/OnnxGraphSurgeon | Med-High |
| C7 | **Rewrite attention layout by editing `Transpose.perm` in place** | how to hand the fused-MHA kernel the layout it wants | 07-Tool/OnnxGraphSurgeon | Med |
| C8 | **Merge a second ONNX file in as a Constant** (`lm_head` weight lifted out of its own ONNX) | joining two exported graphs into one engine | 07-Tool/OnnxGraphSurgeon | Med |
| C9 | **`np.ascontiguousarray` on every `gs.Constant`** | without it TensorRT reads the shape as `(0)` — silent, and the author left a shouting comment | 07-Tool/OnnxGraphSurgeon (trap note) | Med |
| C10 | **Prefill vs decode as two shape regimes on one engine** | the pipeline sets `[B,L]` then `[B,1]`, `L_past` 0 → N; the natural motivation for multiple optimization profiles | ties to §1.6 P5 / `08-Advance/MultiOptimizationProfile` | Med |
| C11 | **Generated-config + Makefile pipeline structure** | every cookbook example is one file; a 4-stage workflow needs staging and caching | 03-Workflow (structure of the new leaf) | Low-Med |
| C12 | **NVTX ranges inside the patched PyTorch model** (`TRAN` green, `LM` blue) | line up torch phases against TRT phases in one Nsight timeline | 07-Tool/nvtx (cross-link) | Low |

### C1 — one tensor instead of 112

`surgeon.py::adjustInputOutput` replaces the 56 KV inputs with a single
`[(56*L_past), B, 32, 128]` tensor plus one `Split` node with `num_outputs=56`, and concatenates the
56 KV outputs into one `[56*(L_past+L), B, 32, 128]` tensor:

```python
tensorInputKV = gs.Variable(..., sDataType, ['(56*L_past)', 'B', 32, 128])
nodeSplit = gs.Node("Split", ..., inputs=[tensorInputKV], outputs=tensorPadKVList,
                    attrs=OrderedDict([('num_outputs', 56)]))
...
graph.inputs  = graph.inputs[:3] + [tensorInputKV]
graph.outputs = [graph.outputs[0], tensorConcat]
```

115 I/O tensors → 6. Per-step that is 112 fewer `set_tensor_address` / `set_input_shape` calls, and
the shape bookkeeping in `main.py` collapses to one line. The idea is not LLM-specific: **any model
with a large fan of same-shaped I/O should be given one packed tensor and a Split/Concat.**

### C2 — the cache grows in place

```python
context.set_tensor_address(lTensorName[3], bufferKVCacheList.data_ptr())  # input
context.set_tensor_address(lTensorName[4], bufferKVCacheList.data_ptr())  # output, same pointer
```

Because the output cache is the input cache with `L` more entries appended along dim 0, input and
output can share one allocation. `backupFile/main-twoBuffer.py` is the earlier ping-pong version
(`bufferKVCacheList = [.., ..]` with `iSource`/`iTarget`) — keeping both makes a good before/after:
the shipped version halves cache memory and drops the swap. Worth pairing with
`05-Plugin/AliasedIOPlugin`, which is the same idea one level down.

### C3 — the engine returns a token, not a logit vector

`surgeon.py::addTail` appends, in ONNX:

```txt
Shape → Gather → Unsqueeze → Sub      # compute the last valid position
Slice(last position)                  # [1, B, 4096]
MatMul(constantLM)                    # [1, B, 130528]   lm_head folded in as a Constant
Softmax → ArgMax → Transpose          # [B, 1]           the next token id
```

with `bKeepLogit` switching the tail back to raw logits when the caller wants its own sampling.
The vocabulary is 130528, so the greedy path moves **2 numbers instead of 261056 per step**. The
transferable lesson is bigger than sampling: *if the host does something fixed to every output,
consider putting it in the graph.*

### C4 — export by patching `forward`

`exportONNX.py` is 14 lines and contains **no export call**:

```python
model = transformers.AutoModel.from_pretrained("pyTorchModel", trust_remote_code=True).eval()
model.chat(tokenizer, query=..., history=[])   # "we will not wait for the response"
```

The real export lives in a patched copy of the model's own `modeling_chatglm.py`
(`backupFile/modeling_chatglm.py`, swapped in by the `Makefile`), which calls `torch.onnx.export`
from inside `forward` — guarded so it fires **on the second call, when `past_key_values` is not
None**, i.e. it captures the *decode* graph with genuine cache inputs rather than a synthetic one,
then exits as soon as both files exist.

That is the answer to "how do I export a model whose `forward` signature I do not control, in the
state it is actually used in". It also makes the 60-input `dynamic_axes` dict a comprehension:

```python
dyDict = {"input_ids": {0: 'B', 1: 'L'}, "position_ids": {0: 'B', 2: 'L'}, "attention_mask": {0: 'B', 2: 'L', 3: 'L'}}
dyDict.update({"input_past_key_%d" % i: {0: 'L_past', 1: 'B'} for i in range(28)})
```

The same patch carries two more useful habits: commented-out blocks that dump every input/output
to `io.npz` for accuracy debugging, and the NVTX ranges of C12.

### C5 — `markGraphOutput`

A 30-line ONNX-GraphSurgeon helper that marks any node's outputs (or inputs, or a chosen index) as
graph outputs and by default **deletes the original outputs**, truncating the graph:

```python
markGraphOutput(graph, ["/Conv"])                     # output tensor of that node
markGraphOutput(graph, ["/Conv"], False, True)        # its inputs (data + weight + bias)
markGraphOutput(graph, ["/TopK"], lMarkOutput=[1])    # its second output
```

This is the practical tool behind "the FP16 engine is wrong somewhere in 400 nodes" — cut the graph
at a node, build, compare, bisect. `07-Tool/OnnxGraphSurgeon/08_isolate_subgraph.py` isolates a
subgraph but does not offer this bisect-the-accuracy-bug workflow, and it pairs naturally with the
Polygraphy `debug reduce` material (§1.3 L1).

### C6 / C7 — the two model-shaped rewrites

`adjustPositionEmbedding` precomputes the rotary tables on the host and Gathers from them:

```python
inv_freq   = 10 ** (-1 / 16 * np.arange(0, 64, 2, dtype=np.float32))
valueTable = (np.arange(nMaxSLPast) ⊗ concat(inv_freq, inv_freq)).reshape(nMaxSLPast, 1, 128)
constantCosTable, constantSinTable = gs.Constant(np.cos(valueTable)), gs.Constant(np.sin(valueTable))
```

`editAttention` then finds each `Softmax` and walks outward (`node.i().i().i()`,
`nodeWhere.i(2).i().i().i().i()`) to retarget the surrounding Transposes to
`[B, 32, L, 128]` / `[B, 32, 128, L]`, dropping a no-op `*1+0` before the `Where`.

Take the **techniques** (host-side constant tables; editing `perm` and re-wiring instead of
inserting new nodes; anchoring the search on a landmark op like `Softmax`), not the code — the
`.i().i().i()` chains are exactly as brittle as they look and are tied to one export of one model.

### C9 — the trap, in the author's own words

```python
# Constant of 1 dimension integer value, MUST use np.ascontiguousarray,
# or TRT will regard the shape of this Constant as (0) !!!
constant0 = gs.Constant("constant0", np.ascontiguousarray(np.array([0], dtype=np.int64)))
```

Worth a one-line note in `07-Tool/OnnxGraphSurgeon` — a non-contiguous array silently becomes a
rank-0 constant.

---

## What NOT to take

+ **The TensorRT-8.6 build recipe is largely illegal under TensorRT 11.** `buildEngine.py` uses
  `create_network(1 << NetworkDefinitionCreationFlag.EXPLICIT_BATCH)` (flag removed in TRT 10),
  `BuilderFlag.FP16` + `OBEY_PRECISION_CONSTRAINTS` and per-layer `set_output_type`
  (**all removed in TRT 11 — networks are strongly typed**), and
  `PreviewFeature.FASTER_DYNAMIC_SHAPES_0805` (gone). If this becomes a cookbook example, the build
  stage is a rewrite, not a port — but that also makes it a good candidate for the
  `trt-strong-typing-migration` story.
+ `config.builder_optimization_level = 5` "to use native TensorRT kernels rather than myelin,
  worse performance but multi-batch support" — an era-specific workaround, and the README admits
  "this will be fixed in the future version". Do not carry the claim forward without re-measuring.
+ `profileList = [builder.create_optimization_profile() for i in range(2)]` while only
  `profileList[0]` is ever added — dead code, but it shows where the author was heading. See C10.
+ Everything ChatGLM-specific: `mask_token_id`/`gmask_token_id`/`bos_token_id` constants, the
  two-row `position_ids` (position + block position), `process_response`'s CJK punctuation
  regexes, the `[Round n]\n问：…\n答：` prompt format, the `.i(2).i().i().i().i()` walks.
+ The 12 GiB engine / 15 GiB peak and the 6B checkpoint: any cookbook version must be re-expressed
  on a tiny model, or it is not runnable in CI.

## Already covered — do not re-propose

Timing cache reuse (`04-Feature/TimingCache`), `hardware_compatibility_level = AMPERE_PLUS`
(`04-Feature/HardwareCompatibility`), `builder_optimization_level`
(`04-Feature/BuilderOptimizationLevel`), NVTX ranges (`07-Tool/nvtx`), constant folding via
Polygraphy (`07-Tool/OnnxGraphSurgeon/06_fold.py`), basic PyTorch→ONNX→TRT
(`03-Workflow/pyTorch-ONNX-TensorRT`).

## Suggested shape of the intake

Not one big example. Two pieces:

1. **`07-Tool/OnnxGraphSurgeon` gains the toolbox** — C5 (`markGraphOutput` bisecting), C6
   (host-side constant tables), C8 (fold a second ONNX in as a Constant), C9 (the contiguity trap).
   All are tiny, self-contained, and testable on the models already in that directory.
2. **One new `03-Workflow` leaf: "a decoder-only model, end to end"** — a *toy* 2-layer
   transformer with a KV cache, carrying C1 (packed cache I/O), C2 (in-place cache, with the
   two-buffer version as the before), C3 (sampling inside the engine), C4 (export from the real
   forward), C10 (prefill/decode profiles), on a model small enough to build in seconds. That is
   the piece the cookbook genuinely lacks: **it has no autoregressive workflow at all.**

The 6B model itself stays where it is — cite `/work/chatglm-6b` as the full-scale reference.

---

## Feasibility on gpt2-medium — **verified 2026-08-28, all four headline techniques transfer**

The 6B model is too big for the cookbook, so the question was whether `gpt2-medium` (the
`00-Data/model/model-large.onnx` already in the tree) can carry the same lessons. Everything below
was **run**, not reasoned about; the export/verification was done on `gpt2` (124M, same architecture
family, 10x faster to iterate) and the shapes were cross-checked against the real gpt2-medium ONNX.

### The KV cache is there, and the I/O explosion is worse than ChatGLM's

| Artifact | I/O tensors |
| -------- | ----------- |
| `00-Data/model/model-large.onnx` = gpt2-medium `decoder_model.onnx` | 2 in (`input_ids`, `attention_mask`) + **49 out** (`logits` + 24 layers x `present.{i}.{key,value}`) — **prefill only, no past inputs** |
| gpt2-medium `decoder_with_past_model.onnx` (same HF folder, 1.63 GB, downloaded and inspected) | **50 in + 49 out = 99 tensors**, `past_key_values.{i}.{key,value}` `[B, 16, past, 64]` |
| the wrapper below | **3 in + 2 out** |

So C1 applies, and harder than on ChatGLM (99 tensors vs 115). gpt2-medium is 24 layers x 16 heads
x 64 head-dim; `decoder_model_merged.onnx` also exists (prefill+decode behind a `use_cache_branch`
`If`), which is a third artifact worth knowing about.

### C4: gpt2 does **not** need the hack — and ChatGLM's hack no longer works anywhere

ChatGLM patched `modeling_chatglm.py` because it is a `trust_remote_code` model whose `forward` you
do not control and whose call is buried inside `model.chat()`. `GPT2LMHeadModel` is a first-class
`transformers` model with a public signature, so none of that applies.

More importantly, **the ChatGLM recipe is dead on modern `transformers` regardless of model**. On
5.16.1 `past_key_values` is typed `transformers.cache_utils.Cache`, and the tuple-of-tuples ChatGLM
handed to `torch.onnx.export` is rejected outright:

```txt
legacy tuple REJECTED: AttributeError 'list' object has no attribute 'get_seq_length'
```

The modern equivalent is a ~15-line wrapper `nn.Module`, and it is **better** than the file patch:
no swapped-in file, no Makefile dance, no "fire on the second call" trick, and the packing of C1
happens in PyTorch instead of in ONNX-GraphSurgeon:

```python
class Gpt2Step(torch.nn.Module):
    def forward(self, input_ids, attention_mask, past_kv):   # past_kv: [LPast, 2N, B, H, D]
        p = past_kv.permute(1, 2, 3, 0, 4)
        cache = DynamicCache()
        for i in range(self.n_layer):
            cache.update(p[2 * i], p[2 * i + 1], i)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         past_key_values=cache, use_cache=True)
        c = out.past_key_values
        present = torch.stack([t for i in range(self.n_layer)
                               for t in (c.layers[i].keys, c.layers[i].values)], 0)
        present = present.permute(3, 0, 1, 2, 4).contiguous()          # [LTotal, 2N, B, H, D]
        next_token = torch.argmax(out.logits[:, -1, :], -1, keepdim=True).to(torch.int32)  # C3
        return next_token, present
```

`torch.onnx.export(..., opset_version=17, dynamo=False)` succeeds. It emits a pile of
`TracerWarning`s (`if not self.is_initialized or self.keys.numel() == 0`, the mask-length checks,
`is_causal = q_length > 1 ...`) which look like the graph is being specialized to the traced
length — **it is not**, verified below. `is_causal` *is* baked, which is why prefill and decode
want separate exports; that is also how ChatGLM did it.

### Verified against PyTorch, at lengths other than the traced one

Traced at `LPast = 3`, then run through onnxruntime and compared with eager PyTorch:

```txt
Lp= 1  onnx_token=  198 torch_token=  198 match=True  present (24,1,12, 2,64) maxdiff=3.62e-05
Lp= 3  onnx_token=  198 torch_token=  198 match=True  present (24,1,12, 4,64) maxdiff=1.81e-05
Lp= 7  onnx_token=   11 torch_token=   11 match=True  present (24,1,12, 8,64) maxdiff=2.29e-05
Lp=16  onnx_token=   68 torch_token=   68 match=True  present (24,1,12,17,64) maxdiff=2.48e-05
```

Token ids identical; the cache differs only by FP32 noise.

### C2 does **not** transfer for free — and that is the interesting part

ChatGLM's cache is `[L, B, 32, 128]`: the sequence axis is **outermost**, so the output cache is the
input cache with rows appended at the end of the allocation, and one buffer can serve both.
HuggingFace's GPT-2 cache is `[B, H, L, D]` — the sequence axis is **dim 2**, so the new entries are
interleaved and a shared buffer is simply wrong.

The fix is to pack sequence-first, `[LPast, 2N, B, H, D]`, at the cost of a `permute` at each end of
the graph. Verified — the output's first `LPast` rows are byte-identical to the input, which is the
precondition for aliasing:

```txt
Lp= 1 token=  198 present (2, 24,1,12,64) prefix-preserved=True
Lp= 5 token=  198 present (6, 24,1,12,64) prefix-preserved=True
Lp=12 token=   68 present (13,24,1,12,64) prefix-preserved=True
```

**This is a better lesson than the original.** ChatGLM's in-place cache looks like a clever trick;
seen next to GPT-2 it is revealed as a *consequence of the cache layout*, and the example can show
both the layout that permits it and the one that does not.

### Verdict

Rewrite on gpt2-medium. C1 / C3 / C4 transfer directly and get simpler; C2 transfers only with the
sequence-first repack, which is worth a case of its own. What is lost is C5-C9 — the
ONNX-GraphSurgeon material — because the wrapper does in PyTorch what ChatGLM had to do with graph
surgery. Those stay a separate `07-Tool/OnnxGraphSurgeon` intake (S6a), sourced from the ChatGLM
code, and the two halves of S6 become genuinely independent.

`transformers` is a new cookbook dependency (checked with `pip install --dry-run`: it pulls only
`typer`, `shellingham`, `annotated-doc` — **it does not touch numpy or tensorrt**).

### gpt2-small vs gpt2-medium — use **small**

|  | gpt2 (small) | gpt2-medium |
| --- | --- | --- |
| layers / heads / hidden | 12 / 12 / 768 | 24 / 16 / 1024 |
| parameters | 124 M | 355 M |
| **vocabulary** | **50257** | **50257** — identical |
| `decoder_with_past_model.onnx` I/O | 26 in + 25 out = **51** | 50 in + 49 out = **99** |
| the same graph through the wrapper | **5** | **5** |
| HF ONNX download | 623 MiB | 1552 MiB |
| exported step graph (measured / estimated) | **652 MB** | ~1.4 GB |
| TensorRT engine, FP32 (measured / estimated) | **476 MiB, built in 11.7 s** | ~1.2 GiB, ~30 s |

Everything the example teaches is unchanged: same architecture, same cache layout, same export path,
and — because the vocabulary is **identical** — the C3 argument (the engine returns one token id
instead of 50257 floats per step) is numerically the same sentence. The only thing lost is that the
headline "99 I/O tensors → 5" becomes "51 → 5".

The example can have both: **case 1 opens the `model-large.onnx` already sitting in `00-Data`**
(gpt2-medium, 2 in + 49 out) to show the explosion at medium scale with zero download, then the
runnable half uses small.

### End to end on small — already proven

Not just exported: parsed, built and run, with `past_kv` and `present_kv` bound to **the same
allocation**.

```txt
engine I/O: ['input_ids', 'attention_mask', 'past_kv', 'next_token', 'present_kv']
prompt   : TensorRT is a
continued:  new type of neural network that can be used to train a
```

So the whole S6b example is de-risked; what remains is packaging it in cookbook style.
