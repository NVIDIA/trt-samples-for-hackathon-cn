# CUDA Graph with Multiple Streams

+ Combine CUDA graph with concurrent streams, where each thread owns its execution context.

+ Steps to run.

```bash
python3 main.py
```

## What this adds over the neighbouring examples

| Example | Covers |
| --- | --- |
| `08-Advance/CudaGraph` | Capture and replay on **one** stream, and one graph per input shape |
| `08-Advance/MultiStream` | **One** execution context issuing into several streams, timed stage by stage |
| **this one** | Several threads running **concurrently**, each with its own context, stream and buffers, with and without a graph |

The gap it fills is the interaction: a graph removes per-launch CPU work, and concurrency is
exactly the regime where that CPU work becomes the ceiling, because every thread is competing for
the same single-threaded launch path.

## You cannot make a TensorRT engine launch bound by adding layers

The obvious way to build this demo is to stack many tiny layers so that issuing them costs more
than running them. **It does not work, and it is worth knowing why before trying.** The builder
fuses any chain of pointwise operations into a single kernel:

| Network | Kernels in the built engine |
| --- | ---: |
| 8 activations | 1 |
| 24 activations | 1 |
| 24 activations of three alternating types | 1 |
| 24 convolution + activation pairs | 1 (one Myelin `ForeignNode`) |

Eliminating those launches is the builder's job, so a workload built that way measures nothing.

What a graph actually removes in TensorRT is the **per-inference CPU cost of `execute_async_v3`**,
which is why `08-Advance/CudaGraph` describes the win as applying to *small engines*: the smaller
the GPU work per call, the larger that fixed CPU cost looms. `build_small_engine` therefore builds
a deliberately small engine and the measurement varies the thing that actually matters here — how
many threads are contending for that one launch path.

## The three cases

### `case_scaling`

Throughput against thread count, with and without a graph. Measured on 2 x RTX PRO 5000 Blackwell,
TensorRT 11.3, 2000 inferences per thread, best of 5 repeats:

| threads | no graph | CUDA graph | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 105,042/s | 160,780/s | 1.53x |
| 2 | 103,704/s | 203,352/s | 1.96x |
| 4 | 54,957/s | 199,414/s | 3.63x |

The shape of this table is the result, not any single number. Without a graph, throughput *falls*
past two threads — the threads are serialising on the launch path, so adding them makes things
worse. With a graph, throughput holds, because one `cudaGraphLaunch` replaces the work they were
contending for. That is why the speedup grows with thread count instead of staying flat.

Read the ratio rather than the absolute numbers: these are Python threads, and the interpreter adds
its own overhead to the no-graph path. Repeats are reduced with `max`, not `mean`, because
contention only ever makes a sample slower.

### `case_context_per_thread`

Each thread gets its own `IExecutionContext`, and both threads' outputs are checked against a
single-threaded reference.

An execution context holds the scratch space for **one inference in flight**. Two threads calling
`execute_async_v3` on the same context do not queue politely; they overwrite each other's
intermediate state. The TensorRT API requires one context per concurrent inference, and the cost is
real — each context carries its own activation memory, so memory grows linearly with thread count.

### `case_graph_replays_addresses`

Two facts that follow from how capture works, both asserted rather than asserted-in-prose:

+ Writing new data into the **same** host buffer does change the next replay's result. The captured
  `cudaMemcpyAsync` holds an address, not a copy of the data.
+ Calling `set_tensor_address` after capture does **not** change where the graph writes. The graph
  recorded kernel arguments at capture time and never consults the context again, so it keeps
  writing to the old buffer.

Together these say: a captured graph owns its buffers for its lifetime. Free or reallocate them and
the graph replays into freed memory, silently.

## Four rules for capturing around a framework

These came out of a kernel-launch-overhead investigation on an AOTInductor model, where a model
loader had to be reconfigured before its inference call could be captured at all. The framework
differs, but the rules transfer to any stack that hides work inside an inference call:

1. **Something in the inference path may be illegal during capture.** Calls that query completion,
   such as `cudaEventQuery`, cannot run inside a capture. In the AOTI case a loader flag controlled
   whether `run()` made such a call; in TensorRT, the one-off work done by the first enqueue after a
   shape change is the equivalent. Either way the fix is the same: get that work to happen *before*
   `cudaStreamBeginCapture`. `Worker.capture` does one full enqueue and a synchronise first.
2. **Removing the obstacle is not the optimisation.** Flipping the flag that was blocking capture
   bought nothing by itself; the gain only arrives once the work is actually captured and replayed.
   Measure after the replay is in place, not after the blocker is removed.
3. **Addresses must be fixed for the life of the graph.** Allocate once, then copy new data into the
   same buffers. This is what `case_graph_replays_addresses` demonstrates.
4. **One instance per thread.** The AOTI loader could not be shared across threads once configured
   for capture; a TensorRT execution context cannot be shared across concurrent inferences either.
   Both cost memory per thread, and both fail quietly rather than loudly when shared.

A fifth, smaller one: **warm up before capturing.** Lazy kernel loading and workspace allocation
inside a framework will otherwise happen during the capture, where they are illegal.

## Measurement caveats

+ Python threads drive the no-graph path through the interpreter, so the speedup here flatters the
  graph a little compared with the same experiment in C++.
+ Discard the first run. Cold-start effects on a launch-bound workload are large.
+ Compare minimum or median, never mean: contention only ever makes a sample slower.
