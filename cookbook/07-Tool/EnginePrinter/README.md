# Engine Printer

+ Print the I/O tables of a TensorRT engine, from the plan file alone.

+ Steps to run.

```bash
python3 main.py
```

| Function | Answers | Available |
| :------- | :------ | :-------- |
| `print_engine_io_information` | the I/O tensors, and the shape range of **every** optimization profile | always |
| `print_engine_information` | the plan header: which TensorRT built it, its hardware-compatibility level, and the device it records | **no — see below** |

## `print_engine_information` is not part of this distribution

Running `main.py` prints:

```txt
    [SKIP] `print_engine_information` is not available in this distribution.
           It reads the plan header field by field, and the serialized layout of a
           TensorRT engine is not part of the public API, so the tool is not shipped.
```

That function reads the plan's header field by field. **The serialized layout of a TensorRT engine
is not part of the public API and is not documented**, and a script that walks it is itself a
description of it, so it is kept out of this repository. The import is optional
(`tensorrt_cookbook/__init__.py` wraps it in a `try`) and nothing else depends on it — the rest of
this example, and the rest of the cookbook, run unchanged without it.

The plan file being opaque is deliberate on TensorRT's part, not an oversight. It is explicitly not
portable across versions, platforms or devices, and nothing in the public API exposes its internals.



## What the public API does and does not give you

`print_engine_io_information` uses only public calls, and covers most of what you would want:

+ every I/O tensor, its dtype, shape and memory location
+ the min / opt / max shape of each tensor **per optimization profile**

Once an engine deserializes, `ICudaEngine` answers a good deal more — `hardware_compatibility_level`,
`num_optimization_profiles`, `num_layers`, `device_memory_size_v2`, `streamable_weights_size`,
`refittable`, `profiling_verbosity`. `trtexec --loadEngine=... --skipInference` prints a summary of
the running device alongside it.

The gap is narrow but real, and worth stating plainly:

+ **`trtexec` reports the *current* device and the *installed* TensorRT**, not what the plan
  recorded. Load a plan built elsewhere and it still prints this machine's compute capability and
  this machine's library version.
+ **`polygraphy inspect model` does not work on an engine under TensorRT 11** as of 0.50.3: it
  reads `engine.device_memory_size`, which was renamed `device_memory_size_v2`, and raises
  `AttributeError`.
+ **When a plan fails to deserialize you get nothing at all.** `deserialize_cuda_engine` returns
  `None`, so every `engine.*` property is unreachable:

```txt
[TRT] [E] IRuntime::deserializeCudaEngine: Error Code 1: Serialization ...
engine.hardware_compatibility_level -> AttributeError: 'NoneType' object has no attribute ...
```

  This is exactly the case where you most want to ask what the plan says, and exactly the case the
  public API cannot answer.

## Two optimization profiles, on purpose

The engine is built with two profiles (batch 1/2/4 and 8/32/64). With only one, the more interesting
half of the I/O table would be empty, and `num_optimization_profiles` would not be worth printing.
