#

## Introduction

+ The example code create a version-compatibility TensorRT engine which can run on Ampere or above GPUs

## Result

+ I tested the script in several scenarios:
  + [x] Build engine on TensorRT 8.6.1.4 with    version-compatibility, run on TensorRT 8.6.1.6.
  + [x] Build engine on TensorRT 8.6.1.6 with    version-compatibility, run on TensorRT 8.6.1.4.
  + [ ] Build engine on TensorRT 8.6.1.4 without version-compatibility, run on TensorRT 8.6.1.6, error information as below.

```text
[E] 6: The engine plan file is not compatible with this version of TensorRT, expecting library version 8.6.1.6 got 8.6.1.4, please rebuild.
[E] 2: [engine.cpp::deserializeEngine::951] Error Code 2: Internal Error (Assertion engine->deserialize(start, size, allocator, runtime) failed. )
Failed building engine!
```
