# 13 - Device Info & GPU Clocks

Query the GPU device information and current clock / power state. This is the
cookbook re-implementation of `trt-engine-explorer`'s `utils/device_info.py`
and the read-only parts of `utils/config_gpu.py`, using **pynvml** instead of
pycuda. It **requires a GPU** and the `pynvml` package.

Engine profiling is most reproducible when the GPU clocks are locked to a fixed
frequency (trex's `config_gpu` does this before profiling). **Locking clocks
needs root privileges**, so this example only *reads* the current and max clocks
and reports whether the GPU appears to be running below its maximum.

Helpers (`tensorrt_cookbook/utils_engine_explorer.py`): `query_device_info`,
`sample_gpu_state`, `get_max_clocks`.

## Running

```bash
python3 main.py   # needs a GPU + pynvml
```

## Output

+ `case_device_info` - name / memory / max clocks per GPU, dumped to `device_info.json`.
+ `case_gpu_state` - current temperature / power / utilization / clocks, and how to lock the clocks.
