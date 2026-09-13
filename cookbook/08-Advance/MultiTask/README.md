# Multi Task

+ Serve several **different** engines at once, combining threads, streams, CUDA graphs and device pinning.

+ Steps to run.

```bash
python3 main.py
```

The cookbook covers the ingredients separately — [`../MultiContext/`](../MultiContext/README.md)
(several contexts of one engine), [`../MultiStream/`](../MultiStream/README.md) (one context, several
streams), [`../MultiDevice/`](../MultiDevice/README.md) (one plan, several GPUs),
[`../CudaGraph/`](../CudaGraph/README.md) (launch overhead replaced by a graph replay). None of them
shows what an inference service actually is: **N different models running concurrently in one
process**, where those features have to be combined and where they interact.

Four models (128–384 channels, 128–256 spatial, 8–24 layers), measured on B200, TensorRT 11.1.0.106,
median of 100 rounds after 20 warm-up:

| Configuration | ms / round | vs seq | minus floor | vs seq |
| ------------- | ---------: | -----: | ----------: | -----: |
| 1 GPU, 1 thread (baseline) | 2.617 | 1.00x | 2.617 | 1.00x |
| 1 GPU, 4 threads + streams | 2.769 | **0.94x** | 2.30 | ~1.1x |
| 1 GPU, 4 threads + CUDA graph | 2.733 | 0.96x | 2.27 | ~1.2x |
| 4 GPUs, threads + graph | **1.221** | **2.14x** | 0.76 | ~3.5x |

## Measure the orchestration floor, or measure Python

The threaded configurations pay a fixed cost — two Python barriers per round — that has nothing to
do with TensorRT. `case_orchestration_floor` runs the identical harness with a **no-op** instead of
an inference. That floor sits under every threaded number above — but it is itself noisy, because it
depends on how the OS schedules four threads through two barriers: across five repeats it ranged
**0.09 – 0.58 ms**, median ~0.3–0.5 ms between runs. The example therefore reports the median with
its spread and warns when the spread is wide, rather than publishing one number that looks precise.
Treat the floor-corrected column as indicative, and the raw column as the measurement.

It also explains an earlier, wrong version of this example. With the tasks sized as they first were
(~0.05 ms each, 0.2 ms per round), threading measured **0.35x** — i.e. concurrency looked like a 3x
*regression*. It was not: the round was 0.2 ms of GPU work under a 0.5 ms floor, so the experiment
was timing Python's barriers. The models here are deliberately large enough (2.6 ms per round) that
the GPU dominates, and the floor is reported next to the result rather than hidden inside it.

**The practical rule: concurrency only pays when per-task work exceeds orchestration cost.** For
microsecond-scale engines, run them sequentially on one thread.

## Concurrency on one GPU buys almost nothing here

0.94x raw, roughly 1.1x with the floor removed. That is not a bug — a single one of these engines already
saturates a B200, so four of them interleaved finish in about the time four of them queued do. The
GPU was never idle waiting for work, which is the only thing extra streams can fix.

## The CUDA graph adds nothing here either, and that is consistent

2.769 → 2.733 ms, within noise. [`../CudaGraph/`](../CudaGraph/README.md) shows graphs paying off
handsomely, but on **small** engines, where per-launch overhead is a large fraction of the work.
These engines are compute-bound: 8–24 convolutions at 256x256 leave launch overhead nowhere to hide.
A graph replaces launch cost, and there is barely any to replace.

## The win is more GPUs

2.14x on 4 GPUs (roughly 3.5x with the floor removed). Spreading distinct models across devices is the one
change that moves the number, because it is the only one that adds hardware rather than rearranging
access to it.

## Three things that fail quietly

1. **A CUDA graph capture must not be the first launch.** Capture records work rather than running
   it, and TensorRT allocates lazily on the first `execute_async_v3` — so capturing a cold context
   either fails or records the allocation into the graph. Warm up, then capture.
2. **`cudaSetDevice` is per-thread.** A worker thread starts on device 0 no matter what the parent
   did. A "pinned" worker that forgets to set its own device still runs, still produces correct
   numbers, and quietly measures one GPU instead of four.
3. **Capture needs a non-default stream.** `cudaStreamBeginCapture` is rejected on the legacy default
   stream, and the cookbook's wrappers default to stream 0, so each task creates its own.

Re-expressed from the idea in the internal `samples_internal/sampleMultiTasks` (Apache-2.0); no code
was taken from it.
