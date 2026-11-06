# GpuTelemetry

+ Record GPU utilisation, memory and power beside a timed run, without being able to break it.

```bash
python3 main.py
```

Adapted from `apps/benchmark/trtmc_benchmark/telemetry.py` in
[TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect) (Apache-2.0).

Every timing example in the cookbook reports latency and nothing else. Latency alone cannot say
whether an engine was compute-bound, whether it fitted in memory, or whether the GPU was busy with
somebody else's work while you measured.

## The design rule worth copying

**Telemetry is best-effort and must never fail the thing it measures.** A missing `nvidia-smi`, a
parse failure, a hung subprocess — each records a string in `telemetry.error` and the measured work
carries on. `case_degrades_instead_of_failing` takes that path on purpose:

```txt
    Work completed under disabled telemetry, result=333328333350000
    samples=0, error="telemetry disabled by mode='off'"
    summary() is still well formed: {'n_sample': 0, 'error': "telemetry disabled by mode='off'"}
```

`summary()` returning a well-formed dict with zero samples is the point. A caller that writes the
summary to a report does not need a branch for "instrumentation was unavailable".

Three smaller rules, each easy to get wrong:

+ **Daemon thread**, so a stuck sampler can never hold the process open.
+ **`self._stop.wait(interval)` rather than `time.sleep(interval)`**, so `__exit__` stops the loop
  immediately instead of after the current interval elapses.
+ **Bounded join** on exit (`interval + 1 s`, minimum 2 s). A hung `nvidia-smi` must not become a
  hung benchmark; if the thread outlives the join, that is recorded as an error too.

## Two things the measurements taught us

**`n_sample` counts rows, not polls.** One `nvidia-smi` call returns one line per visible GPU, so on
this 4-GPU host every poll appends 4 samples. Worse, the statistics then mix the GPUs — an idle card
drags `memory_used_mib.min` down to 4 MiB while the busy one reads 1558 MiB:

```txt
    rows=236 (59 polls x 4 GPU), device=NVIDIA B200
    memory_used_mib           min=     4.00  mean=   337.48  max=  1558.00
```

`summary()` therefore reports `n_poll` and `n_device` separately. Set `CUDA_VISIBLE_DEVICES` and the
sampler passes `--id` to `nvidia-smi`, which is the only way to get statistics about *your* GPU.

**The effective sampling period is the interval plus the subprocess cost.** Asking for 20 ms over a
1.68 s run produced 21 polls, not 84: launching `nvidia-smi` costs tens of milliseconds and that
lands between intervals. Treat the interval as a lower bound.

## Does the instrument change the number?

`case_sampling_cost_stays_outside` runs the same 20 000 inferences at two sampling rates:

| interval | polls | ms/iteration |
| --- | --- | --- |
| 20 ms | 21 | 0.0841 |
| 200 ms | 7 | 0.0840 |

A 3x difference in poll count moved per-iteration latency by **0.1%**, which supports the claim that
the sampler sits beside the timed region rather than inside it. Read it as a sanity check and not a
guarantee: it is one unrepeated pair on a shared GPU, and the case says so in its own output.

The loop has to be long enough for the two rates to differ at all. At 300 iterations it finishes
inside one 200 ms interval, both settings collect a single poll, and the comparison is vacuous —
which is what the first version of this case did.

## Where the interesting numbers are

`case_build_is_where_the_memory_goes` points the same instrument at an engine build:

```txt
    Engine build took 6754.01 ms, engine is 12.66 MiB
    rows=236 (59 polls x 4 GPU), ... gpu_utilization_percent max=10.00, memory_used_mib max=1558.00
```

Inference on this MNIST network is 0.08 ms and invisible to a 20 ms sampler. The build is nearly
seven seconds of real work. If you only ever instrument the inference loop you will never see it.

Measured on TensorRT 11.3.0.99, 4x B200, 2026-10-12.
