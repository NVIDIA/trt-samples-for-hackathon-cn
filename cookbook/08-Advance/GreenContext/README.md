# Green Context

+ Give a TensorRT engine a fixed slice of one GPU's SMs, from inside the process.

+ Steps to run.

```bash
python3 main.py
```

A green context (CUDA 12.4+) partitions the SMs of a GPU and hands out a stream bound to the
partition; everything launched on that stream is confined to those SMs. TensorRT needs no API for
it — `execute_async_v3(green_stream)` is the whole integration — which is exactly why the two
surprises below are easy to walk into.

All numbers measured on **B200 (148 SM)**, TensorRT 11.1.0.106, CUDA 13.3, at 1965 MHz / 26 C.
Re-measured 2026-09-06 after the platform moved from H100 PCIe (114 SM); every number below
changed, and one conclusion changed with them.

## The partition is real, and free

The engine is a chain of eight 1024x1024 matrix multiplies, chosen to be SM bound:

| Stream | Latency | vs whole GPU | SM ratio |
| ------ | ------- | ------------ | -------- |
| default (148 SM) | 0.050 ms | 1.00x | 1.00x |
| green, 16 SM | 0.222 ms | 4.42x | 9.25x |
| green, 32 SM | 0.123 ms | 2.45x | 4.62x |
| green, 64 SM | 0.072 ms | 1.44x | 2.31x |
| green, 148 SM | 0.049 ms | **0.99x** | 1.00x |

The last row matters: a partition containing every SM costs nothing, so the mechanism itself adds
no overhead. The scaling is sub-linear (16 SM is 9.3x fewer SMs but only 4.4x slower) because a
smaller partition still gets the whole L2 and memory system.

Partitions are not arbitrary: `minSmPartitionSize` and `smCoscheduledAlignment` are **still both 8**
on B200, so `cuDevSmResourceSplitByCount` rounds the request. Ask the returned resource what you
got — the SM count changed with the platform, the granularity did not.

## What it is for: the noisy neighbour

A latency-critical engine, with a throughput-hungry engine running beside it in the same process:

| Setup | median | p95 |
| ----- | ------ | --- |
| alone | 0.031 ms | 0.037 ms |
| background job, both on the default stream | 0.061 ms (**1.95x**) | 0.096 ms (**2.60x**) |
| background job, green 32 SM / 116 SM | 0.035 ms (1.13x) | 0.037 ms (**0.99x**) |

**Both halves of this got better on B200, and the second one is now essentially perfect.** On H100
contention cost 4.12x median / 7.51x p95 and the partition recovered to 1.86x / 2.41x; on B200
contention costs only 1.95x / 2.60x and the partition recovers to 1.13x / **0.99x** — a partitioned
latency job is indistinguishable from one running alone. Two effects push the same way: 148 SM
leaves more room for both jobs, and 32 of 148 is a smaller slice than 32 of 114 was.

This is the experiment MIG is usually sold with, done **without** MIG: same process, no root, no
host configuration, no container restart, and the partitions can be created and destroyed at will.
See [`../MIG/README.md`](../MIG/README.md) for what MIG buys that this does not (hardware-level
memory and bandwidth isolation, separate fault domains, cross-container assignment).

## The hole: auxiliary streams escape the partition

TensorRT runs independent branches of a network on **auxiliary streams**, and those are created
from the current context, not from the green one. The background engine below is pinned to its own
82 SM, the latency job to a disjoint 32 SM, and yet:

| Background engine built with | latency median | p95 |
| ---------------------------- | -------------- | --- |
| `max_aux_streams = 0` | 0.035 ms (1.09x) | 0.043 ms (1.27x) |
| `max_aux_streams = 4` | 0.051 ms (1.57x) | 0.092 ms (**2.70x**) |
| default (`-1`, TensorRT decides) | 0.053 ms (1.63x) | 0.095 ms (**2.78x**) |

The absolute numbers shrank with the SM count but **the leak did not go away**: p95 still more than
doubles the moment auxiliary streams are allowed, on a partition that is supposed to be disjoint.

The default is `-1`, so **nobody has to ask for this to happen**. If the isolation matters, build
the engines that live in a partition with `max_aux_streams = 0` and pay for it in intra-engine
concurrency — or measure the p95, which is where the leak shows up first.

## The free 8%: build where you will run

Building the engine while the green context is current makes it faster on that partition:

```txt
primary context: cudaGetDeviceProperties.multiProcessorCount=148, cuCtxGetDevResource=148
green context  : cudaGetDeviceProperties.multiProcessorCount=148, cuCtxGetDevResource=32
built on the whole GPU (1), run on 32 SM:   0.123 ms
built on the whole GPU (2), run on 32 SM:   0.123 ms
built inside the partition, run on 32 SM:   0.113 ms
build-to-build spread of two identical builds: 0.000 ms; partition effect: 0.010 ms
```

Two builds under identical conditions differ by 0.000-0.001 ms, so the 0.010 ms (**8%**) is an
order of magnitude above the noise and not an artefact — three consecutive runs gave 0.123/0.123/0.113,
0.123/0.123/0.113 and 0.124/0.123/0.113 ms. The two control builds are in the example for exactly
this reason. **The effect is real but smaller than the 19% measured on H100**, which is what you
would expect: with 148 SM the whole-GPU tactic choice is less wrong for a 32 SM partition than it
was with 114.

The mechanism is worth understanding, because the obvious explanation is wrong: TensorRT cannot
*ask* how large the partition is. `cudaGetDeviceProperties.multiProcessorCount`, which is what it
reads, still reports 148 inside the green context; only the driver-level `cuCtxGetDevResource`
knows about the 32. What adapts is the **tactic search**, which is empirical: candidate kernels are
timed in whatever context is current, so a build done inside the partition measures the partition's
real behaviour and picks different winners.

This is the measured version of the argument in [`../MIG/README.md`](../MIG/README.md) — build on
the profile you deploy on.

## Lifetime trap

An engine built while a green context is current owns CUDA resources belonging to that context.
Destroying the context first makes TensorRT's destructors fail with
`Error Code 1: Cuda Runtime (In deallocate ...)` and then takes the process down with a SIGSEGV
**at exit**, far from the mistake. Either let the context outlive the TensorRT objects (what case 4
does) or destroy the objects first. Cases 1-3 can free their partitions normally, because their
engines live in the primary context and only the *stream* comes from the partition.

## Related

+ [`../MIG/README.md`](../MIG/README.md) — the hardware-partitioning alternative, and why this
  directory is a note instead of an example.
+ [`../MultiStream/`](../MultiStream/README.md), [`../MultiContext/`](../MultiContext/README.md) —
  sharing one GPU without partitioning it.
+ [`../../04-Feature/AuxStream/`](../../04-Feature/AuxStream/README.md) — what `max_aux_streams`
  does when nothing is partitioned.
