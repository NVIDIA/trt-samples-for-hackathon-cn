# 11-TensorRT-Model-Connect

+ Patterns adapted from [NVIDIA/TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect),
  a production deployment stack built on TensorRT.

TRTMC turns a Hugging Face checkpoint into a runnable bundle in two commands, across 136 model
families. It is not a teaching repository — it is what the API looks like when somebody has to ship
with it. The examples here are the parts of that experience that transfer to anyone using TensorRT
directly, rebuilt small enough to read in one sitting.

Everything in this directory is adapted under Apache-2.0, the same licence as the cookbook, with the
upstream file named at the top of each source file.

| Directory | What it shows | Upstream |
| --- | --- | --- |
| [`GraphTransform/`](GraphTransform/README.md) | Change a network built by code you do not own, at the last moment before it is compiled | `core/builder/tensorrt_model_connect/graph_transform.py` |
| [`GpuTelemetry/`](GpuTelemetry/README.md) | Record GPU utilisation, memory and power beside a timed run, without being able to break it | `apps/benchmark/trtmc_benchmark/telemetry.py` |
| [`ParameterizedPlugin/`](ParameterizedPlugin/README.md) | One plugin that hosts any registered kernel, selected by a string at build time | `core/runtime/byok/`, `core/builder/tensorrt_model_connect/byok.py` |
| [`Glossary/`](Glossary/README.md) | Every term in the cookbook, in plain words, pointing at the example that shows it | `website/docs/getting-started/glossary.md` |

```bash
cd GraphTransform && python3 main.py
cd GpuTelemetry   && python3 main.py
cd ParameterizedPlugin && python3 main.py
# Glossary is a reading page with no main.py
```

## The common thread

Each of these answers a question the cookbook had not asked, because an example that builds one
engine and prints one number never hits it:

+ **What if I do not own the builder?** `GraphTransform`. A library builds the network; you need one
  more output on it.
+ **What if latency is not the whole answer?** `GpuTelemetry`. Was the GPU even busy? Did it fit?
  Who else was on the card?
+ **What if I need fifty kernels?** `ParameterizedPlugin`. Fifty plugin classes, or one plugin and
  fifty registered functions.
+ **What does this word mean?** `Glossary`. Sixty directories of examples and, until now, no page
  that defines "optimization profile".

## What was left upstream

The survey that produced this directory was removed on 2026-10-12; its conclusions are compressed
into the ecosystem table in [`../99-Todo/README.md`](../99-Todo/README.md). Short version of what
was examined and rejected: TRTMC's family architecture is a deliberate
*anti*-abstraction stance ("duplicate model-specific code by design") that is right for 136
independently-owned models and wrong for a cookbook whose point is shared helpers; and its bundle
format, dev toolkit and evidence reporting solve fleet problems the cookbook does not have.

Measured on TensorRT 11.3.0.99, B200, 2026-10-12.
