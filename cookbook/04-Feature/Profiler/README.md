# Profiler

+ `IProfiler` and, on top of it, a latency report an application can produce for itself.

## Steps to run

```bash
python3 main.py

cd C++
make build
./main.exe
```

## The two cases in `main.py`

+ `case_normal` — the `IProfiler` interface on its own, and the effect of
  `IExecutionContext.enqueue_emits_profile`. When it is `True` (the default) every enqueue is
  reported; when it is `False`, only the one enqueue following a `report_to_profiler()` call is.
+ `case_latency_report` — the measurement built on top of it, described below.

`C++/main.cpp` is the same measurement from C++, on the same network, so the two can be compared
directly on one machine.

## Why measure in-process at all

[`../../07-Tool/trtexec/parse_export_json.py`](../../07-Tool/trtexec/README.md) already computes
percentiles — but only *after* trtexec has written its JSON. An application that embeds TensorRT has
no trtexec and no JSON; it has to measure itself. `samples/common/sampleReporting.{h,cpp}` in
TensorRT-OSS is the code trtexec uses, and both files here re-express the parts a deployed
application actually needs, against the public API only.

Measured on B200 (idle), TensorRT 11.0.0.114, the cookbook's MNIST network at batch 4 (the largest
its optimization profile allows), 200 iterations after 20 warm-up.

## One number hides where the time goes

```txt
      metric |       min       max      mean    median       p90       p95       p99       cv
---------------------------------------------------------------------------------------------
         H2D |    0.0071    0.0149    0.0079    0.0079    0.0083    0.0084    0.0099    8.93%  ms
     compute |    0.0479    0.0568    0.0499    0.0499    0.0508    0.0511    0.0536    2.18%  ms
         D2H |    0.0093    0.0179    0.0101    0.0099    0.0108    0.0109    0.0170   11.10%  ms
     latency |    0.0656    0.0769    0.0678    0.0678    0.0690    0.0704    0.0755    2.60%  ms
     enqueue |    0.0302    0.0380    0.0310    0.0308    0.0313    0.0319    0.0363    3.20%  ms
```

The finding here is the last row. `enqueue` is the host-side cost of `execute_async_v3` alone, and
at 0.0308 ms it is **62% of the 0.0499 ms of compute it launches**. The CPU is very nearly the
bottleneck: this engine is launch-bound, and no faster kernel would help it —
[`../../08-Advance/CudaGraph`](../../08-Advance/CudaGraph/README.md) is what would. A single
end-to-end latency number cannot say any of that, because `enqueue` overlaps the device work and
never appears in it.

The copies are 26% of the latency, so they are worth knowing about but are not the story on this
network. On a network with large inputs they easily become it; the point of splitting the
measurement is that you do not have to guess which case you are in.

Three metrics, three different tools:

| Metric | Measured with | Why |
| ------ | ------------- | --- |
| H2D / compute / D2H | separate `cudaEvent` pairs on the stream | device-side, ordered with the work |
| enqueue | `time.perf_counter` / `std::chrono` around the enqueue call | host-side cost, invisible to CUDA events |
| per-layer | `IProfiler` | attributes device time to layers |

The host buffers are **pinned** (`cudaHostAlloc`). With pageable memory the driver stages the
transfer through its own buffer, and H2D/D2H would be measuring that instead. This is also why
`case_latency_report` deserializes its own engine rather than calling `tw.setup()` — the wrapper's
host buffers are ordinary numpy arrays.

## Percentiles and the coefficient of variation

The mean of a latency distribution with a tail is not a latency anyone experiences, so the summary
reports min / max / mean / median / p90 / p95 / p99 plus the **coefficient of variation** (standard
deviation as a percentage of the mean). The coefficient of variation is the one that says whether
the run is worth quoting at all: at 2.60% these numbers are stable, and a run at 30% is telling you
about the machine, not the engine — see the thermal-throttling note in
[`../../08-Advance/ContextParallelism/`](../../08-Advance/ContextParallelism/README.md) for what
that looks like when it goes wrong.

## Aggregate the per-layer profile with the median

`report_layer_time` is called once per layer per execution, so 50 executions of an 11-layer engine
produce 550 samples. Summing them measures how long you profiled; averaging them lets one slow first
iteration dominate. The upstream `LayerProfile::median` takes the median per layer, and so does
`MedianProfiler` here — note the contrast with `CookbookProfiler`, which prints one line per call
and is right for `case_normal` but the wrong shape for measurement.

```txt
 median ms     share  layer
    0.0148     16.6%  MatrixMultiplication1_myl0_7
    0.0129     14.4%  MatrixMultiplication2_myl0_8
    0.0108     12.0%  Convolution2_myl0_4
    0.0088      9.8%  Convolution1_myl0_2
    0.0068      7.6%  __myl_Topk_myl0_10
    0.0068      7.6%  __myl_MaxrSubExpSumDiv_myl0_9
    0.0067      7.5%  Pooling1_myl0_3
    0.0066      7.4%  Pooling2_myl0_5
    0.0065      7.3%  __myl_Resh_myl0_6
    0.0064      7.1%  __myl_Move_myl0_1
    0.0025      2.8%  __mye324_0_myl0_0
----------------------------------------------------------------------------------------------------
    0.0895           (sum of per-layer medians)
```

The two fully-connected layers cost more than either convolution here — at batch 4 on a 28x28
image, the convolutions are small and the classifier head is not.

**The sum does not match the un-profiled compute time (0.0895 vs 0.0499 ms), and that is expected**:
attaching a profiler serialises the layers so each can be timed. Per-layer numbers are for finding
the expensive layer, never for quoting a total. That is also why the two measurements above use two
separate execution contexts.

Note also that the layer names are the *engine's*, not the network's. The names set by
`build_mnist_network_trt` survive with a `_myl0_N` suffix marking the Myelin-fused region they
landed in, but the `__myl_*` and `__mye*` entries are fusions with no counterpart in the network at
all — `__myl_MaxrSubExpSumDiv_myl0_9` is the whole softmax collapsed into one kernel. Reconciling
those back to source layers is what [`../../07-Tool/trex/`](../../07-Tool/trex/README.md) is for.

## Setting the profiler to `None` does not detach it

Profiling is a one-way switch for the lifetime of an `IExecutionContext`.

```txt
Error Code 3: API Usage Error (Parameter check failed, condition: (profiler) != nullptr)
```

**The python binding fails more quietly than the C++ one.** `context.profiler = None` raises no
python exception; the assignment statement looks like it worked. The only sign is that `Error Code
3` line from the TensorRT logger, which is easy to lose in build output. Reading the attribute back
still returns the old profiler, and it keeps being called:

```txt
    After `context.profiler = None`, context.profiler is MedianProfiler, not None
    and it was still called 11 more times by the next execution.
```

So do not rely on detaching. Profile in a context you are about to destroy, or keep a separate
un-profiled context for the hot path.

## Related

+ [`../../07-Tool/trtexec/`](../../07-Tool/trtexec/README.md) — the same statistics from the command
  line, plus `parse_export_json.py` for the JSON it exports.
+ [`../../07-Tool/trex/`](../../07-Tool/trex/README.md) — engine-level analysis of the exported layer
  information.
+ [`../ProfilingVerbosity/`](../ProfilingVerbosity/README.md) — how much layer detail the engine
  keeps in the first place.
