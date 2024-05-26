# PythonPlugin

+ The same as BasicExample, but we build and run the engine with plugin totally in Python script.

+ These examples show 5 ways (using cuda-python, cupy, torch, triton, numba packages respectively) to make it.

+ We keep a cuda-python example (add_scalar_cuda_python-V2-deprecated.py) to use deprecated class `IPluginV2DynamicExt`.

+ This example is too simple to show the performance differences among the libraries.

+ TODO:
  + Remove the redundant memory copy in torch / triton example, which need a solution of wrapping a pointer as a torch.tensor.
  + Get rid of using cupy, so remove the examples with suffix "-using-cupy".
  + Fix numba example, now I get error like below.

```txt
[ERROR] Exception thrown from enqueue() LinkerError: [222] Call to cuLinkAddData results in CUDA_ERROR_UNSUPPORTED_PTX_VERSION
ptxas application ptx input, line 9; fatal   : Unsupported .version 8.4; current version is '8.3'
```

## Steps to run

```shell
python3 add_scalar_cuda_python.py
python3 add_scalar_cupy.py
python3 add_scalar_numba.py
python3 add_scalar_torch.py
python3 add_scalar_triton.py
```
