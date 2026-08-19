# Distributed Inference

+ Data-parallel inference with Torch-TensorRT: one engine per GPU, the batch split across them.

+ Steps to run.

```bash
python3 main.py
```

Upstream `examples/distributed_inference/` covers two different things under one heading, and only
one of them runs here:

| | what it is | status |
| --- | --- | --- |
| **Data parallel** | the whole model on every GPU, the batch split | measured below |
| **Tensor parallel** | one model *split across* GPUs, NCCL collectives inside the graph | **blocked here** |

Tensor parallel needs the TRT-LLM plugin library, and importing `torch_tensorrt` in this container
reports:

```txt
CUDA 13 is not currently supported for TRT-LLM plugins.
Please install pytorch with CUDA 12.x support
```

The TensorRT-level equivalent is **not** blocked — [`../../../08-Advance/ContextParallelism/`](../../../08-Advance/ContextParallelism/README.md)
performs the same collectives through `IDistCollective` without going near TRT-LLM.

Measured on 4x B200, TensorRT 11.1.0.106, `torch_tensorrt` 2.14.0a0, 16 × `Linear(4096, 4096)` FP16,
256 samples per GPU:

| configuration | batch | ms/round | samples/s | speed-up |
| --- | ---: | ---: | ---: | ---: |
| 1 GPU | 256 | 0.169 | 1,513,912 | 1.00x |
| 4 GPUs, data parallel | 1024 | 0.218 | **4,704,236** | **3.11x** |

All four replicas agree bit-for-bit on identical input (`max |diff| = 0.000e+00`), which is checked
before any throughput number is quoted.

## Each GPU needs its own compilation

An engine is device-specific — see [`../../../08-Advance/MultiDevice/`](../../../08-Advance/MultiDevice/README.md),
where engine *bytes* move across devices but an `ICudaEngine` does not. So a module compiled on
device 0 cannot simply be `.to("cuda:1")`; every replica compiles its own from the same weights.

## How you measure decides what you measure

This example got 3.11x only after the measurement was fixed. Two earlier versions reported
**0.79x** and **0.65x** — i.e. data parallelism looking like a *regression* on 4 GPUs:

+ The first synchronised the workers with two Python barriers **per iteration**. At 0.079 ms of GPU
  work per round, the barriers cost more than the inference, so the experiment measured threading.
+ Enlarging the model to 0.239 ms/round was not enough; the floor is roughly 0.4–0.5 ms
  (measured directly in [`../../../08-Advance/MultiTask/`](../../../08-Advance/MultiTask/README.md)).

The fix was to measure **throughput** rather than per-round latency: each worker runs all its
iterations back to back, and the wall clock covers the whole run, so the barrier is paid once
instead of twice per round. The single-GPU baseline is measured the same way, or the comparison
would be unfair in the other direction.

The general rule: when per-iteration work approaches the cost of the synchronisation used to
orchestrate it, the benchmark stops being about the GPU.

## Related

+ [`../../../08-Advance/ContextParallelism/`](../../../08-Advance/ContextParallelism/README.md) —
  tensor/context parallelism at the TensorRT level, which is what tensor-parallel Torch-TRT would
  have shown.
+ [`../../../08-Advance/MultiTask/`](../../../08-Advance/MultiTask/README.md) — several *different*
  models at once, and the orchestration floor measured directly.
+ [`../../../08-Advance/MultiDevice/`](../../../08-Advance/MultiDevice/README.md) — why each device
  needs its own engine, and how fast they can be loaded in parallel.
