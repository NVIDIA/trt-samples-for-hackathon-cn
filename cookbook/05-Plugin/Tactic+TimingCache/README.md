# Tactic+TimingCache

+ The same as BasicExample, but we use our own tactics and timing-cache in the plugin.

+ To understand timing-cache generally, refer to `02-API/TimingCache`.

+ In this example we create a network with two same plugin layers, and build the engine two times.

+ For each plugin layer, we provide two custom tactics for trying.

+ So we can see TensorRT profiling the two custom tactics and reuse the result cross layers and buildings by timing cache.

+ We can use VERBOSE log and add `-DDEBUG` into compilation commands to see the process of TensorRT building.

+ Steps to run.

```bash
make test
```
