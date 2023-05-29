#

## Introduction

+ The example code create a hardware-compatibility TensorRT engine which can run on Ampere or above GPUs

## Result

+ I tested the script in several scenarios:
  + Build engine on A100 GPU without hardware-compatibility, run on A100 GPU: 4.13489 ms (GPU compute time by trtexec)
  + Build engine on A100 GPU with    hardware-compatibility, run on A100 GPU: 4.13599 ms
  + Build engine on A100 GPU without hardware-compatibility, run on A10  GPU: can not run, error information as below
  + Build engine on A100 GPU with    hardware-compatibility, run on A10  GPU: 3.43140 ms
  + Build engine on A10  GPU without hardware-compatibility, run on A10  GPU: 3.70880 ms

+ Error information while building on A100 (CC=8.0) and running on A10 (CC=8.6) without hardware-compatibility

```text
[E] Error[6]: The engine plan file is generated on an incompatible device, expecting compute 8.6 got compute 8.0, please rebuild.
[E] Error[2]: [engine.cpp::deserializeEngine::951] Error Code 2: Internal Error (Assertion engine->deserialize(start, size, allocator, runtime) failed. )
[E] Engine deserialization failed
[E] Got invalid engine!
[E] Inference set up failed
```
