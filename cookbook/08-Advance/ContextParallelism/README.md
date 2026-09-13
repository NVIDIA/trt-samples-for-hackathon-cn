# Context Parallelism

+ Split one attention model across GPUs along the sequence axis, with the collectives inserted into the ONNX graph.

+ Steps to run.

```bash
python3 main.py    # One configuration (sequence 16384 by default)
python3 sweep.py   # Sequence 2048 -> 65536, to find where 2 GPUs start to pay off
```

[`../../02-API/Layer/DistCollective/`](../../02-API/Layer/DistCollective/README.md) shows the
collective operations one at a time on a toy tensor. This is the same machinery on a real workload:
a self-attention block sharded across 2 GPUs so each one owns half of the sequence.

All numbers measured on B200 (148 SM), TensorRT 11.1.0.106, polygraphy 0.50.3, float16,
8 heads x 64 head-dim, batch 1. **Read the thermal note at the bottom before trusting any latency
you measure yourself on this machine.**

## No `mpirun`

The upstream sample (`samples/python/attention_mdtrt`) is launched with `mpirun -np 2`. This one is
not: like `02-API/Layer/DistCollective`, `main.py` re-launches *itself* once per rank and passes the
NCCL unique id through a file, so it stays a plain `python3 main.py` and the unified test runner
needs no special case. `mpi4py` is never imported. On a machine with fewer than 2 GPUs it prints a
skip line and exits 0.

## The pipeline

| Step | What happens |
| ---- | ------------- |
| `case_build_model` | Build the single-device attention ONNX with the onnx-graphsurgeon layer API |
| `case_shard_model` | `polygraphy multi-device shard` rewrites it using `hint.json` |
| `case_single_device` | Build + run the unsharded model on one GPU — the numerical reference |
| `case_context_parallel` | One process per rank; each attaches a NCCL communicator and runs the sharded model |

Sharding adds 8 ONNX nodes, 4 of them `DistCollective`:

| Name | Operation | Reduce | Purpose |
| ---- | --------- | ------ | ------- |
| `shard_1` | `reduce_scatter` | `max` | Split the input sequence across ranks |
| `shard_5` | `all_gather` | – | Every rank needs the full K to attend over the whole sequence |
| `shard_11` | `all_gather` | – | Same for V |
| `shard_15` | `all_gather` | – | Reassemble the full output |

TensorRT layers go 57 → 65. The only runtime API involved is
`IExecutionContext.set_communicator(capsule)` — TensorRT drives the collectives itself.

**Why every rank is fed the whole input.** The leading collective is `ReduceScatter` with
`reduce_op=max`, and all ranks hold identical data, so the reduction is the identity and the
operation degenerates into "scatter the sequence". Together with the trailing `AllGather` this means
the sharded engine has the **same I/O contract** as the unsharded one — same input shape, same
output shape — which is what makes the two directly comparable.

## Where 2 GPUs start to pay off

Attention is O(sequence²) while the collectives are O(sequence), so context parallelism is a loss on
short inputs and a win on long ones. `sweep.py` measures the crossover instead of asserting it:

| Sequence | 1 GPU | 2 GPUs | Speed-up | rel. diff | SM clock | Verdict |
| -------: | ----: | -----: | -------: | --------: | -------: | ------- |
| 2048 | 0.077 ms | 0.154 ms | 0.50x | 0 | 1965/1965 | single GPU |
| 4096 | 0.110 ms | 0.220 ms | 0.50x | 2.7e-03 | 1965/1965 | single GPU |
| 8192 | 0.279 ms | 0.308 ms | 0.90x | 2.9e-03 | 1965/1965 | single GPU |
| 16384 | 0.869 ms | 0.649 ms | **1.34x** | 3.4e-03 | 1965/1965 | context parallel |
| 32768 | 2.809 ms | 1.812 ms | **1.55x** | 4.2e-03 | 1897/1965 | context parallel |
| 65536 | 10.884 ms | 5.692 ms | **1.91x** | 2.8e-03 | 1822/1935 | context parallel |

Below 16384 the two K/V `AllGather`s cost more than the attention work they save. The speed-up
approaches 2x only as the quadratic term takes over — at 65536 the single-GPU time is growing ~3.9x
per doubling, i.e. essentially quadratic, while the 2-GPU time grows ~3.1x.

## Three things the upstream sample does not tell you

### `hint.json` needs `polygraphy_class` keys

The upstream README documents a clean, minimal hint file. That file **does not work**: polygraphy
deserializes it through its own JSON machinery, which dispatches on a `polygraphy_class` key, and
without it the tool stops at

```txt
[!] Provided JSON cannot be decoded into a ShardHints.
```

The working file also needs `k_seq_len_idx` / `v_seq_len_idx` / `kv_rank` at the top level, which the
README's example omits entirely. `main.py` writes the full form and comments why.

### Static reshape shapes make the sharded model unparseable

Write the sequence length as a constant in the Reshape targets and the *unsharded* model is fine,
but the sharded one is rejected:

```txt
Error Code 4: Shape Error (reshape changes volume. Reshaping [8192,1,512] to [16384,1,8,64])
```

After `ReduceScatter` each rank holds half the tokens, so any hard-coded sequence length is wrong on
every rank. The upstream sample solves this with a `Shape`→`Gather`→`Unsqueeze`→`Concat` chain per
reshape; writing the sequence dimension as `-1` is 12 nodes cheaper and says the same thing.

### The sharding tool renames the graph output

`output` becomes `shard_14`. Anything that binds tensors by name downstream has to be told.

## Numerics: sometimes bit-identical, and that is not luck

The relative difference against the single-GPU reference is either ~3e-03 or **exactly 0**, and
which one you get depends on the sequence length (see the table: 2048 gives 0). This is not a flaky
comparison. `ReduceScatter(max)` over identical inputs and `AllGather` are exact data movement, so
the sharded model performs the *same* arithmetic on the same values; whether the result is bitwise
equal then comes down to whether the builder picked the same kernels for both engines. Expect either,
assert neither — `main.py` asserts a relative bound, not equality.

## Thermal throttling will destroy these measurements

This is not a footnote, it is the reason the numbers above are usable. Running the **same** engine
(identical 18 layers, identical 322 MiB device memory) at sequence 32768:

| GPU | temperature | latency over 5 rounds | SM clock |
| --- | ----------- | --------------------- | -------- |
| one already used for benchmarking | 88-91 °C | 3.6 → 11.0 ms | 1432 → **352 MHz** |
| an untouched one | 36 → 54 °C | 2.769 – 2.815 ms (spread 1.7%) | ~1897 MHz |

`clocks_throttle_reasons.active` reads `0x20`, SW thermal slowdown, at 437 W against a 1000 W limit —
so it is heat, not power. An earlier version of this sweep, run without any guard, reported
**111.8 ms** at sequence 65536 where the correct answer is **10.9 ms**: a 10x error that looks exactly
like a real measurement. It also moved the apparent crossover from 16384 to 32768 and produced a
nonsensical 14.45x "speed-up" on one row.

So `main.py`:

+ picks the **coolest** GPUs rather than hard-coding devices 0 and 1,
+ waits for them to fall below 70 °C before timing anything (and says so loudly if they do not),
+ prints the SM clock and temperature next to every latency,

and `sweep.py` refuses to publish a row quietly, warning by name about any row measured below
1500 MHz. If you re-run this on a machine that cannot cool 2 GPUs, the warning is the point.

## Related

+ [`../../02-API/Layer/DistCollective/`](../../02-API/Layer/DistCollective/README.md) — the five
  collective operations individually, and the same NCCL-id-via-file launcher.
+ [`../MultiDevice/`](../MultiDevice/README.md) — engine bytes are portable across devices, engines
  are not.
+ [`../../05-Plugin/NcclPlugin/`](../../05-Plugin/NcclPlugin/README.md) — doing the communication
  yourself in a plugin instead of letting TensorRT own it.
