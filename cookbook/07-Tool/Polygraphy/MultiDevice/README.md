# Polygraphy - Multi-Device mode

+ Steps to run (introduction is included in the script).

```bash
./main.sh
```

+ `polygraphy multi-device shard` rewrites a single-device ONNX model into a multi-device one, and
  `polygraphy template shard-hints` writes the JSON that drives it. Both are new in Polygraphy
  0.50. The workflow mirrors [`../Plugin/`](../Plugin/README.md):
  `shard-hints` -> review / edit the JSON by hand -> `shard` (or `--one-shot` to skip the file).

+ **Everything in `main.sh` is a pure ONNX rewrite and runs on a single GPU.**
  Executing the sharded models needs one process per rank and is **not** covered, see case 06 of
  the script for what is left to do and where the multi-GPU plumbing lives.

## What the two modes actually do

|  | CP (context parallel) | TP (tensor parallel) |
| --- | --- | --- |
| splits | the **sequence** | the **weights** |
| pattern it looks for | attention body: `MatMul(Q,K) -> Softmax -> MatMul(.,V)` | SwiGLU MLP (keyed off a `Sigmoid`), or an `AttentionPlugin` node |
| output files | one | **one per rank**, `<name>_tp<N>_rank<i>.onnx` |
| initializers | unchanged, every rank holds a full copy | sliced: `w_gate`/`w_up` `(8,16) -> (8,8)` by column, `w_down` `(16,8) -> (8,8)` by row |
| collectives inserted | 6 (`reduce_scatter` in, `all_gather` for K/V and the output) | 1 (`all_reduce` on the output) |

Measured on the toy block of [`build_transformer_block.py`](./build_transformer_block.py)
(10 nodes, `B=1 S=4 H=8 I=16`): CP gives 16 nodes with the same 4 initializers, TP gives 11 nodes
per rank with every MLP matrix halved.

## Traps worth knowing

+ **A model without the pattern shards silently into a copy.** Case 05 runs the whole flow on
  `00-Data/model/model-trained.onnx` (MNIST): the hints file is written with
  `"attention_layers": []`, the sharder reports success, and the output has exactly the same 12
  nodes and 0 collectives as the input. Nothing warns. **Check `attention_layers` in the JSON, not
  the exit code.**
+ **The rank count comes from `--nb-rank`, not `--gpus`.** `--gpus 2` alone leaves the default
  `--nb-rank 1`, and TP then writes a single `_tp1_rank0` file that is a copy of the input.
  `--gpus` only fills `dist_collectives.group_size`.
+ **`-o` must end in `.json`** for `template shard-hints`, and the check happens *after* the model
  is loaded and analysed (`[!] Output file must be a json`).
+ The sharded graph contains `DistCollective` nodes, which are NCCL collectives rather than
  standard ONNX. onnxruntime cannot run them, so the single-device model is the only thing that
  gives a free numerical reference.

## Related

+ [`../../../05-Plugin/NcclPlugin/`](../../../05-Plugin/NcclPlugin/README.md) — the NCCL side of
  multi-GPU inference in TensorRT.
+ [`../../../08-Advance/MultiDevice/`](../../../08-Advance/MultiDevice/README.md) — engine bytes
  can be shared across devices, an `ICudaEngine` cannot.
+ [`../Plugin/`](../Plugin/README.md) — the same "generate a config, edit it, apply it" shape.
