# Multi Device

+ Example to show `engine_bytes` can be shared cross devices, but `engine` can not.

+ Steps to run.

```shell
python3 main.py             # engine bytes are portable across devices, engines are not
cd C++ && make build && ./main.exe   # loading one plan onto N GPUs: sequential vs std::async
```

## Loading one plan onto N GPUs (`C++/main.exe`)

Because an `ICudaEngine` cannot be shared across devices, a multi-GPU process must deserialize the
same plan once per device. On a large plan that is real startup cost, and the obvious fix is to do
the N deserializations concurrently. Whether that helps is not obvious: host-side work (parsing,
allocation, kernel tables) parallelises, an H2D weight copy over a shared root complex does not.

Measured on 4x B200, TensorRT 11.1.0.106, an 18 MiB plan (8 convolutions, 256 channels), best of 3:

| | wall time | per-device |
| --- | --: | --: |
| sequential | 4.9 ms | 1.2 ms each |
| parallel (`std::async`) | **2.4 ms** | 1.4–2.1 ms each |

**2.02x on 4 GPUs — 50% of linear.** Worth doing, but not the 4x the shape of the problem suggests:
each individual deserialization gets *slower* under contention (1.2 ms → 1.4–2.1 ms), which is the
signature of a shared resource rather than four independent loads.

### Three traps, all of which fail quietly

1. **`cudaSetDevice` is per-thread.** A thread started by `std::async` inherits nothing and begins on
   device 0. Forget to set the device inside the worker and all N engines land on one GPU — the run
   still succeeds, and the "parallel" number is meaningless.
2. **`std::async` without a launch policy may not start a thread at all.** The default policy is
   allowed to defer the work until `.get()`, which would silently turn this into the sequential
   case. `std::launch::async` forces a real thread.
3. **Identical weights make the plan too small to measure.** Pointing all 8 convolutions at one
   weight buffer gives a **2 MiB** plan instead of 18 MiB, because TensorRT deduplicates identical
   weights in the engine — and 2 MiB deserializes too fast to time. (The same trap bit the engine
   size comparison in `07-Tool/OnnxVisualization`, where `nn.TransformerEncoder` deep-copies one
   layer N times.)

Re-expressed from the idea in the internal `samples_internal/deserializeTimer`; no code was taken
from it.
