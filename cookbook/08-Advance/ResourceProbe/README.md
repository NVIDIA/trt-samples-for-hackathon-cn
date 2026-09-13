# Resource Probe

+ How much GPU memory can you actually use, and in what order must you give it back?

+ Steps to run.

```bash
python3 main.py
```

Two questions that decide whether a deployment survives its first week, and that neither the API nor
`nvidia-smi` answers directly. Both are re-expressed as **concepts only** from the internal
`testing/tools/{allocation_checker,trt_shutdown_test}/`, whose code is proprietary — nothing was
copied. Measured on B200, TensorRT 11.1.0.106.

## Reported free vs usable

`cudaMemGetInfo` reports free memory; what you can allocate is a different number. The only reliable
answer is empirical — allocate in chunks until it fails:

```txt
GPU 0: 181993 MiB free of 182619 MiB total, per cudaMemGetInfo
allocated 16384 MiB in 64 chunks of 256 MiB
driver overhead on top of the request: 0 MiB
after freeing everything: 181993 MiB free (started at 181993 MiB)
```

On this datacentre part the gap is **zero** — `cudaMemGetInfo` is trustworthy here. That is worth
knowing as a baseline, and it is why the technique matters more on embedded parts, where the driver
reserve is a large fraction of a small total and the same probe returns a very different answer. The
probe is capped at 16 GiB and stops 2 GiB short of exhaustion so it stays polite on a shared machine;
it asserts it gave everything back.

## Fragmentation

Allocate 64 blocks, free every other one, then ask for a single block the size of everything freed —
the state a long-running server reaches, and the reason a model that "fits" fails to load after a
few hours.

```txt
freed every other block: 2048 MiB returned, 179945 MiB reported free
a single 2048 MiB allocation still succeeded -- the allocator coalesced the holes
```

Another honest negative: the CUDA allocator coalesced the holes, so this simple pattern does not
reproduce fragmentation. Reproducing it takes adversarial sizing, which is itself the point — casual
fragmentation is not the usual cause of a failed load.

## Release order, and the error that exits 0

In Python, destroying the objects in the wrong order is usually survivable, because reference
counting keeps them alive until the last reference goes:

```txt
reverse of creation (correct)    survived
runtime first (wrong)            survived
```

The interesting failure is different, and it is the one worth guarding against. When TensorRT
objects outlive the CUDA context they were created under — here forced with an early
`cudaDeviceReset()` — the teardown breaks. **But the symptom is not stable.** Three identical runs:

| run | exit code | destructor errors logged |
| --- | --- | ---: |
| 1 | 0 | 5 |
| 2 | **-11 (SIGSEGV)** | 0 |
| 3 | **-11 (SIGSEGV)** | 0 |

```txt
Error 201 destroying event '0x45058990'. In ~MyelinGraphContext at .../graphContext.cpp:101
Error Code 1: Cuda Runtime (In deallocate at .../defaultAllocator.cpp:91)
```

Both outcomes are bad and **neither is reliably detectable on its own**: a harness that checks only
the exit status misses run 1, and one that checks only the log misses runs 2 and 3. If you run
TensorRT under a custom allocator, a green context, or anything that tears down CUDA state, check
*both* — and do not conclude from one clean run that the ordering is correct.

This also cost this example a flaky test. The first version asserted the specific outcome it happened
to observe (errors logged, exit 0), and failed intermittently in CI for the same reason the bug is
hard to catch. It now asserts only that the teardown broke, by one route or the other.

## Related

+ [`../GreenContext/`](../GreenContext/README.md) — the same lifetime rule with teeth: there,
  destroying the context first takes the process down with a SIGSEGV at exit.
+ [`../../04-Feature/GPUAllocator/`](../../04-Feature/GPUAllocator/README.md) — taking over allocation
  entirely, which makes the ordering question yours to answer.
+ `tests/check_cpp_ownership.py` — the static check that enforces this ordering in the C++ examples.
