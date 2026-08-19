# pyTorch-KVCache-ONNX-TensorRT

+ An **autoregressive** model end to end: PyTorch → ONNX → TensorRT, with a KV cache.

The cache is the interesting part here, not the kernels.

+ Steps to run.

```bash
python3 main.py
```

Measured on B200, TensorRT 11.1.0.106, `transformers` 5.16.1, on `gpt2` (124M). A run writes
~2.2 GB of ONNX + engines into this directory.

**The checkpoint is not downloaded by this example.** It is read from `00-Data/model/gpt2/`, which
is populated once by a separate step:

```bash
cd <PathToCookbook>/00-Data
python3 get-model-gpt2.py     # ~528 MB
```

If that directory is missing, `main.py` prints the command above and exits 0 rather than reaching
for the network. This is the cookbook-wide convention — see
[`../../00-Data/README.md`](../../00-Data/README.md); no example fetches anything while it runs,
so every one of them works offline.

The **ONNX files are cached** `make`-style — delete them and the next run re-exports them. The
**engines are rebuilt every run** (**16.4 s** each), on purpose: a serialized plan is only valid on
the compute capability it was built on, so a cached one carried to another machine fails at load
with `Error Code 6: The engine plan file is generated on an incompatible device, expecting compute
10.0 got compute 9.0`. Saving 16 s is not worth handing that message to the next reader.

Every other `03-Workflow` example converts a model that is called **once**. Call one per generated
token, with a cache in tow, and three new problems appear. The techniques here come from a
hand-written ChatGLM-6B pipeline (see [`../../99-Todo/chatglm-6b.md`](../../99-Todo/chatglm-6b.md)),
reduced to a model that builds in seconds.

## 1. The problem: the cache becomes a wall of graph I/O

`00-Data/model/model-large.onnx` is already in the tree — it is gpt2-**medium**'s
`decoder_model.onnx`, and it is prefill-only:

```txt
inputs    2: ['input_ids', 'attention_mask']
outputs  49: ['logits', 'present.0.key', 'present.0.value'] ... (24 layers x key/value)
the matching decode graph adds 48 past inputs -> 50 in + 49 out = 99 I/O tensors
```

**99 tensors, each needing a `set_input_shape` and a `set_tensor_address`, every single token.**
That is what the rest of this example removes.

## 2. Pack the cache in PyTorch, not in ONNX-GraphSurgeon

`transformers` hands the cache around as a `Cache` object — one `[B, H, L, D]` tensor per layer per
key/value — and `torch.onnx.export` faithfully turns each into its own graph I/O. The fix is to wrap
the model in a `Module` whose signature is **flat tensors**:

```python
class Gpt2Step(torch.nn.Module):
    def forward(self, input_ids, attention_mask, past_kv):   # past_kv: [LPast, 2N, B, H, D]
        past = past_kv.permute(1, 2, 3, 0, 4)
        cache = DynamicCache()
        for i in range(self.n_layer):
            cache.update(past[2 * i], past[2 * i + 1], i)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            past_key_values=cache, use_cache=True)
        c = output.past_key_values
        present = torch.stack([t for i in range(self.n_layer)
                               for t in (c.layers[i].keys, c.layers[i].values)], 0)
        present = present.permute(3, 0, 1, 2, 4).contiguous()
        next_token = torch.argmax(output.logits[:, -1, :], -1, keepdim=True).to(torch.int32)
        return next_token, present
```

```txt
model-gpt2-step.onnx: 2538 nodes
    in  ('input_ids',      ['B', 'L'])
    in  ('attention_mask', ['B', 'LTotal'])
    in  ('past_kv',        ['LPast', 24, 'B', 12, 64])
    out ('next_token',     ['B', 1])
    out ('present_kv',     ['LTotal', 24, 'B', 12, 64])
-> 5 I/O tensors instead of 51 (gpt2) or 99 (gpt2-medium)
```

Six lines of PyTorch replace a `Split`/`Concat` rewrite over ~100 tensors in ONNX-GraphSurgeon.

**The export emits alarming `TracerWarning`s** — `if not self.is_initialized or self.keys.numel() == 0`,
the attention-mask length checks, `is_causal = q_length > 1 ...` — which read as "the graph is being
frozen to the traced length". It is not. Traced at `L_past = 3`, checked against eager PyTorch:

```txt
L_past= 1: TensorRT token    11, PyTorch token    11, match=True
L_past= 7: TensorRT token    11, PyTorch token    11, match=True
L_past=16: TensorRT token    68, PyTorch token    68, match=True
```

`is_causal` **is** baked, which is why this graph is the *decode* step only; a prefill graph wants
its own export. Note `dynamo=False`: the TorchScript exporter is deprecated in torch 2.9+, but the
dynamo exporter takes a different path through the `Cache` object and is not what was verified here.

## 3. One allocation for both the past and the present

Because the engine writes `L_past + 1` entries where it read `L_past`, **appending at the end of the
allocation**, the input and the output binding can be the same pointer:

```python
context.set_tensor_address("past_kv",    cache.data_ptr())   # the same
context.set_tensor_address("present_kv", cache.data_ptr())   # allocation
```

```txt
engine 476 MiB, built in 16.4 s
engine I/O: ['input_ids', 'attention_mask', 'past_kv', 'next_token', 'present_kv']
one cache allocation of 18 MiB serves both bindings for all 17 steps
prompt    : TensorRT is a
continued :  new type of neural network that can be used to train a neural
```

No copy, no ping-pong, and the cache is allocated once at `MAX_LENGTH` rather than growing. This is
[`../../05-Plugin/AliasedIOPlugin/`](../../05-Plugin/AliasedIOPlugin/README.md)'s idea one level up:
there a plugin declares aliased I/O, here the *caller* simply points two bindings at one buffer.

## 4. …but only because of the layout, and that is the real lesson

Aliasing is legal only if the engine **appends** — the first `L_past` elements of the output must be
the input, byte for byte. That is a property of the memory layout, not a TensorRT feature:

```txt
[2N, B, H, L, D] (as transformers stores it): first 92160 values equal the input? False
[L, 2N, B, H, D] (what this example exports): first 92160 values equal the input? True
```

HuggingFace keeps the sequence on axis 3, so new entries are **interleaved** and a shared buffer is
silently wrong — right shape, right dtype, no error, wrong numbers. Repacking sequence-first costs
one `permute` at each end of the graph and makes the append real.

ChatGLM-6B's cache is `[L, B, 32, 128]` and its pipeline shares the buffer without comment; seen
next to GPT-2 that is not a clever trick but a consequence of how the model happened to store its
cache. **Check the layout before reaching for the trick.**

## 5. Sampling belongs in the graph

The engine can return the token id instead of the logit vector — `argmax` is one node:

| output | per token | what the host does |
| ------ | --------- | ------------------ |
| `next_token` `(1, 1)` int32 | **0.597 ms** | copy 4 bytes |
| `logit` `(1, 50257)` float32 | 0.684 ms | copy 196 KiB, then `argmax` |

**196 KiB less to move per token, 1.15x on the whole step** (the ratio moves run to run). The margin
is modest here only because gpt2 is small enough that a decode step is 0.60 ms; the *bytes* saved
are the same for any model with this vocabulary, so the bigger the batch and the shorter the step,
the more it matters. The general rule is worth more than the number: **if the host does the same
fixed thing to every output, the graph can do it instead.**

Both variants are built from the same wrapper via `b_return_logit`, so the comparison is like for
like.

## Notes

+ The engine is FP32: TensorRT 11 is strongly typed, so precision is decided by the ONNX. Export the
  model in half precision to get an FP16 engine (and roughly half the 476 MiB).
+ One optimization profile covers `L_past` in `0 .. MAX_LENGTH - 1`. Prefill and decode are the two
  shape regimes an LLM engine really has — see
  [`../../08-Advance/MultiOptimizationProfile/`](../../08-Advance/MultiOptimizationProfile/README.md)
  for serving both from one engine.
+ `transformers` is required. It pulls only `typer` / `shellingham` / `annotated-doc` and does not
  touch `numpy` or `tensorrt`.

## Related

+ [`../pyTorch-ONNX-TensorRT/`](../pyTorch-ONNX-TensorRT/README.md) — the same road for a model that
  is called once.
+ [`../../05-Plugin/AliasedIOPlugin/`](../../05-Plugin/AliasedIOPlugin/README.md) — aliased I/O
  declared by a plugin instead of arranged by the caller.
+ [`../../99-Todo/chatglm-6b.md`](../../99-Todo/chatglm-6b.md) — the 6B pipeline these techniques
  come from, including the ONNX-GraphSurgeon half that this example does not need.
